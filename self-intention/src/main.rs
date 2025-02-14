use anyhow::Ok;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{loss, Linear, Module, Optimizer, Sequential, VarMap, SGD};

struct SelfAttention {
    d: u32, // Embedding size
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
}

impl SelfAttention {
    fn new(d: u32) -> Self {
        let d = d;
        let w_q = Linear::new(d, d);
        let w_k = Linear::new(d, d);
        let w_v = Linear::new(d, d);

        Self { d, w_q, w_k, w_v }
    }
}

impl Module for SelfAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);

        let qk = q.matmul(&k.transpose(1, 0));
        let qk = qk / D::from_f32((self.d as f32).sqrt());
        let qk = softmax(&qk, 1);

        qk.matmul(&v)
    }
}
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();

    Ok(())
}
