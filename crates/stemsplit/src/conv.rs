//! The U-Net's convolutions on the CPU: im2col of one output row and one
//! matrix product per row, the rows in parallel.
//!
//! candle 0.9's CPU `conv2d` builds im2col tiles of 512 pixels from a
//! channels-last copy it makes on one thread, and its `conv_transpose2d` is a
//! scalar loop; together they ran this network at ~11 GFLOP/s (docs/timeline.md,
//! 0.2.0). Here the im2col copy is row slices, one row's columns stay in cache
//! (larger bands of rows were slower), and a stride-2 transposed convolution is
//! four ordinary 2x2 convolutions, one per output parity. The matrix product is
//! still candle's (gemm), so no `unsafe`.

use anyhow::{ensure, Result};
use candle_core::{Device, IndexOp, Tensor};
use rayon::prelude::*;

/// One item's activations, `[channel][row * w + col]`: rows are frequency bins, cols frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Act {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Act {
    /// `a`'s channels, then `b`'s (`Tensor::cat` on the channel axis).
    pub fn cat(mut a: Act, b: &Act) -> Act {
        debug_assert_eq!((a.h, a.w), (b.h, b.w));
        a.data.extend_from_slice(&b.data);
        a.c += b.c;
        a
    }
}

/// Input index along one axis: `out * stride + offset`, one offset per tap.
#[derive(Clone, Debug)]
struct Taps {
    stride: usize,
    offsets: Vec<isize>,
}

/// `[cout, cin * row taps * col taps]` against the input's taps; no bias.
struct Product {
    w: Tensor,
    cout: usize,
    rows: Taps,
    cols: Taps,
}

impl Product {
    /// `[cout][oh * ow]`.
    fn run(&self, x: &Act, oh: usize, ow: usize) -> Result<Vec<f32>> {
        let rows = (0..oh)
            .into_par_iter()
            .map(|y| self.row(x, y, ow))
            .collect::<Result<Vec<_>>>()?;
        let plane = oh * ow;
        let mut out = vec![0.0f32; self.cout * plane];
        out.par_chunks_mut(plane).enumerate().for_each(|(co, dst)| {
            for (y, r) in rows.iter().enumerate() {
                dst[y * ow..][..ow].copy_from_slice(&r[co * ow..][..ow]);
            }
        });
        Ok(out)
    }

    /// Output row `y`, `[cout][ow]`.
    fn row(&self, x: &Act, y: usize, ow: usize) -> Result<Vec<f32>> {
        let (nr, nc) = (self.rows.offsets.len(), self.cols.offsets.len());
        let k = x.c * nr * nc;
        let mut col = vec![0.0f32; k * ow];
        let w = x.w as isize;
        for ci in 0..x.c {
            let plane = &x.data[ci * x.h * x.w..][..x.h * x.w];
            for (a, &dy) in self.rows.offsets.iter().enumerate() {
                let iy = (y * self.rows.stride) as isize + dy;
                if iy < 0 || iy >= x.h as isize {
                    continue; // padding: stays zero
                }
                let src = &plane[iy as usize * x.w..][..x.w];
                for (b, &dx) in self.cols.offsets.iter().enumerate() {
                    let d = &mut col[((ci * nr + a) * nc + b) * ow..][..ow];
                    if self.cols.stride == 1 {
                        // Columns with 0 <= out + dx < w, as one slice.
                        let lo = (-dx).max(0);
                        let hi = (w - dx).min(ow as isize);
                        if lo < hi {
                            d[lo as usize..hi as usize]
                                .copy_from_slice(&src[(lo + dx) as usize..(hi + dx) as usize]);
                        }
                    } else {
                        for (xo, v) in d.iter_mut().enumerate() {
                            let ix = (xo * self.cols.stride) as isize + dx;
                            if ix >= 0 && ix < w {
                                *v = src[ix as usize];
                            }
                        }
                    }
                }
            }
        }
        let col = Tensor::from_vec(col, (k, ow), &Device::Cpu)?;
        Ok(self.w.matmul(&col)?.flatten_all()?.to_vec1()?)
    }
}

fn bias_of(b: Option<&Tensor>, cout: usize) -> Result<Vec<f32>> {
    Ok(match b {
        Some(b) => b.to_vec1()?,
        None => vec![0.0; cout],
    })
}

fn finish(mut v: Vec<f32>, bias: &[f32], plane: usize, relu: bool) -> Vec<f32> {
    v.par_chunks_mut(plane).zip(bias).for_each(|(ch, &b)| {
        for x in ch.iter_mut() {
            *x += b;
            if relu && *x < 0.0 {
                *x = 0.0;
            }
        }
    });
    v
}

/// `candle_nn::Conv2d` (square kernel, no groups or dilation).
pub struct Conv {
    product: Product,
    bias: Vec<f32>,
    k: usize,
    stride: usize,
    pad: usize,
}

impl Conv {
    pub fn new(m: &candle_nn::Conv2d) -> Result<Self> {
        let cfg = m.config();
        ensure!(
            cfg.groups == 1 && cfg.dilation == 1,
            "grouped or dilated conv"
        );
        let (cout, cin, k, kw) = m.weight().dims4()?;
        ensure!(k == kw, "square kernels only");
        let offsets: Vec<isize> = (0..k).map(|i| i as isize - cfg.padding as isize).collect();
        let taps = Taps {
            stride: cfg.stride,
            offsets,
        };
        Ok(Conv {
            product: Product {
                w: m.weight().reshape((cout, cin * k * k))?.contiguous()?,
                cout,
                rows: taps.clone(),
                cols: taps,
            },
            bias: bias_of(m.bias(), cout)?,
            k,
            stride: cfg.stride,
            pad: cfg.padding,
        })
    }

    pub fn forward(&self, x: &Act, relu: bool) -> Result<Act> {
        let o = |n: usize| (n + 2 * self.pad - self.k) / self.stride + 1;
        let (oh, ow) = (o(x.h), o(x.w));
        let v = self.product.run(x, oh, ow)?;
        Ok(Act {
            c: self.product.cout,
            h: oh,
            w: ow,
            data: finish(v, &self.bias, oh * ow, relu),
        })
    }
}

/// `candle_nn::ConvTranspose2d` with kernel 4, stride 2, padding 1: the output
/// doubles each axis. Output row `2m + p` takes input rows `m` and `m - 1`
/// (kernel rows 1, 3) when `p` is 0, rows `m + 1` and `m` (kernel rows 0, 2)
/// when it is 1; columns likewise. So each output parity is a 2x2 convolution.
pub struct ConvUp {
    phases: Vec<(usize, usize, Product)>,
    bias: Vec<f32>,
    cout: usize,
}

const UP_TAPS: [([u32; 2], [isize; 2]); 2] = [([1, 3], [0, -1]), ([0, 2], [1, 0])];

impl ConvUp {
    pub fn new(m: &candle_nn::ConvTranspose2d) -> Result<Self> {
        let cfg = m.config();
        let (cin, cout, k, kw) = m.weight().dims4()?;
        ensure!(
            (
                k,
                kw,
                cfg.stride,
                cfg.padding,
                cfg.output_padding,
                cfg.dilation
            ) == (4, 4, 2, 1, 0, 1),
            "only kernel 4, stride 2, padding 1"
        );
        let dev = m.weight().device();
        let mut phases = Vec::new();
        for (py, (ky, dy)) in UP_TAPS.iter().enumerate() {
            for (px, (kx, dx)) in UP_TAPS.iter().enumerate() {
                // [cin, cout, 2, 2] -> [cout, cin * 4], taps in (ci, a, b) order.
                let w = m
                    .weight()
                    .index_select(&Tensor::new(&ky[..], dev)?, 2)?
                    .index_select(&Tensor::new(&kx[..], dev)?, 3)?
                    .permute((1, 0, 2, 3))?
                    .contiguous()?
                    .reshape((cout, cin * 4))?;
                let taps = |offsets: &[isize; 2]| Taps {
                    stride: 1,
                    offsets: offsets.to_vec(),
                };
                phases.push((
                    py,
                    px,
                    Product {
                        w,
                        cout,
                        rows: taps(dy),
                        cols: taps(dx),
                    },
                ));
            }
        }
        Ok(ConvUp {
            phases,
            bias: bias_of(m.bias(), cout)?,
            cout,
        })
    }

    pub fn forward(&self, x: &Act, relu: bool) -> Result<Act> {
        let (oh, ow) = (2 * x.h, 2 * x.w);
        let mut out = vec![0.0f32; self.cout * oh * ow];
        for (py, px, p) in &self.phases {
            let v = p.run(x, x.h, x.w)?;
            out.par_chunks_mut(oh * ow)
                .zip(v.par_chunks(x.h * x.w))
                .for_each(|(dst, src)| {
                    for m in 0..x.h {
                        let row = &mut dst[(2 * m + py) * ow..][..ow];
                        for (n, s) in src[m * x.w..][..x.w].iter().enumerate() {
                            row[2 * n + px] = *s;
                        }
                    }
                });
        }
        Ok(Act {
            c: self.cout,
            h: oh,
            w: ow,
            data: finish(out, &self.bias, oh * ow, relu),
        })
    }
}

/// Item `b` of a `[B, C, H, W]` tensor.
pub fn act_of(x: &Tensor, b: usize) -> Result<Act> {
    let (_, c, h, w) = x.dims4()?;
    Ok(Act {
        c,
        h,
        w,
        data: x.i(b)?.flatten_all()?.to_vec1()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{Conv2dConfig, ConvTranspose2dConfig, Module, VarBuilder, VarMap};

    fn max_diff(a: &Tensor, b: &Act) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.data.len());
        a.iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    // Odd sizes, so every edge and padding case shows.
    fn input(c: usize, h: usize, w: usize) -> Tensor {
        Tensor::rand(-1f32, 1f32, (1, c, h, w), &Device::Cpu).unwrap()
    }

    #[test]
    fn conv_matches_candle_for_every_shape_the_net_uses() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        for (i, (cin, cout, k, stride, padding)) in
            [(3, 5, 3, 1, 1), (4, 6, 4, 2, 1), (5, 3, 1, 1, 0)]
                .into_iter()
                .enumerate()
        {
            let cfg = Conv2dConfig {
                padding,
                stride,
                ..Default::default()
            };
            let m = candle_nn::conv2d(cin, cout, k, cfg, vb.pp(format!("c{i}"))).unwrap();
            let x = input(cin, 70, 36);
            let want = m.forward(&x).unwrap();
            let got = Conv::new(&m)
                .unwrap()
                .forward(&act_of(&x, 0).unwrap(), false)
                .unwrap();
            assert_eq!(want.dims()[2..], [got.h, got.w]);
            assert!(max_diff(&want, &got) < 1e-5, "conv {i}");
        }
    }

    #[test]
    fn transposed_conv_matches_candle() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = ConvTranspose2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let m = candle_nn::conv_transpose2d(6, 4, 4, cfg, vb.pp("up")).unwrap();
        let x = input(6, 37, 18);
        let want = m.forward(&x).unwrap();
        let got = ConvUp::new(&m)
            .unwrap()
            .forward(&act_of(&x, 0).unwrap(), false)
            .unwrap();
        assert_eq!(want.dims()[2..], [got.h, got.w]);
        assert!(max_diff(&want, &got) < 1e-5);
    }
}
