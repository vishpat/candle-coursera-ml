use anyhow::Result;
use candle_core::Result as CandleResult;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_nn::{linear, Linear, Module, VarMap};

struct SelfAttention {
    d: usize, // Embedding size
    masked: bool,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

fn get_mask(size: usize, device: &Device) -> CandleResult<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> CandleResult<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl SelfAttention {
    fn new(d: usize, masked: bool, vb: VarBuilder) -> Result<Self> {
        let d = d;

        let w_q = linear(d, d, vb.pp("w_q"))?;
        let w_k = linear(d, d, vb.pp("w_k"))?;
        let w_v = linear(d, d, vb.pp("w_v"))?;

        Ok(Self {
            d,
            masked,
            w_q,
            w_k,
            w_v,
        })
    }
}

impl Module for SelfAttention {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        let sims = q.matmul(&k.transpose(1, 0)?)?;
        let scale = Tensor::new((self.d as f32).sqrt(), sims.device())?;
        let mut scaled_sims = sims.broadcast_div(&scale)?;
        if self.masked {
            let mask = get_mask(scaled_sims.dims()[0], scaled_sims.device())?;
            scaled_sims = masked_fill(&scaled_sims, &mask, f32::NEG_INFINITY)?;
        }
        let attn_pct = softmax(&scaled_sims, 1)?;
        let attn_scores = attn_pct.matmul(&v)?;
        Ok(attn_scores)
    }
}

struct MultiHeadAttention {
    heads: usize,
    d: usize,
    masked: bool,
    w_qs: Vec<Linear>,
    w_ks: Vec<Linear>,
    w_vs: Vec<Linear>,
}

impl MultiHeadAttention {
    fn new(heads: usize, d: usize, masked: bool, vb: VarBuilder) -> Result<Self> {
        let w_qs = (0..heads)
            .map(|_| linear(d, d, vb.pp("w_q")))
            .collect::<CandleResult<_>>()?;
        let w_ks = (0..heads)
            .map(|_| linear(d, d, vb.pp("w_k")))
            .collect::<CandleResult<_>>()?;
        let w_vs = (0..heads)
            .map(|_| linear(d, d, vb.pp("w_v")))
            .collect::<CandleResult<_>>()?;

        Ok(Self {
            heads,
            d,
            masked,
            w_qs,
            w_ks,
            w_vs,
        })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut attns = Vec::with_capacity(self.heads);
        for i in 0..self.heads {
            let q = self.w_qs[i].forward(x)?;
            let k = self.w_ks[i].forward(x)?;
            let v = self.w_vs[i].forward(x)?;

            let sims = q.matmul(&k.transpose(1, 0)?)?;
            let scale = Tensor::new((self.d as f32).sqrt(), sims.device())?;
            let mut scaled_sims = sims.broadcast_div(&scale)?;
            if self.masked {
                let mask = get_mask(scaled_sims.dims()[0], scaled_sims.device())?;
                scaled_sims = masked_fill(&scaled_sims, &mask, f32::NEG_INFINITY)?;
            }
            let attn_pct = softmax(&scaled_sims, 1)?;
            let attn_scores = attn_pct.matmul(&v)?;
            attns.push(attn_scores);
        }
        let attns = Tensor::stack(&attns, 0)?;
        Ok(attns)
    }
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let self_attn = SelfAttention::new(2, true, vs)?;

    let encoding_matrix = Tensor::new(
        vec![
            vec![1.16 as f32, 0.23 as f32],
            vec![0.57 as f32, 1.36 as f32],
            vec![4.41 as f32, -2.16 as f32],
        ],
        &device,
    )?;

    let attn = self_attn.forward(&encoding_matrix)?;
    println!("{}", attn);
    Ok(())
}
