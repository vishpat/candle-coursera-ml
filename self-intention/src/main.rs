use anyhow::Ok;
use anyhow::Result;
use candle_core::Var;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_nn::{linear, Linear, Module, Optimizer, Sequential, VarMap, SGD};

struct SelfAttention {
    d: usize, // Embedding size
    scale: Tensor,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

impl SelfAttention {
    fn new(d: usize, vb: VarBuilder) -> Result<Self> {
        let d = d;
        let scale = Tensor::new((d as f32).sqrt(), vb.device())?;
        let w_q = linear(d, d, vb.pp("w_q"))?;
        let w_k = linear(d, d, vb.pp("w_k"))?;
        let w_v = linear(d, d, vb.pp("w_v"))?;

        Ok(Self {
            d,
            scale,
            w_q,
            w_k,
            w_v,
        })
    }
}

impl SelfAttention {
    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        let qk = q.matmul(&k.transpose(1, 0)?)?;

        let qk = qk.broadcast_div(&self.scale)?;
        let qk = softmax(&qk, 1)?;

        Ok(qk.matmul(&v)?)
    }
}
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let self_attn = SelfAttention::new(4, vs)?;

    Ok(())
}
