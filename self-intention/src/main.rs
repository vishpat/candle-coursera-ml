use anyhow::Ok;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::VarBuilder;
use candle_nn::{linear, Linear, Module, VarMap};

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
        println!("x: {:?}", x);
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        println!("q: {:?}, k: {:?}", q, k);
        let qk = q.matmul(&k.transpose(1, 0)?)?;

        let qk = qk.broadcast_div(&self.scale)?;
        let qk = softmax(&qk, 1)?;

        println!("qk: {:?}, v: {:?}", qk, v);
        Ok(qk.matmul(&v)?)
    }
}
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let self_attn = SelfAttention::new(2, vs)?;

    let encoding_matrix = Tensor::new(
        vec![
            vec![1.16 as f32, 0.23 as f32],
            vec![0.57 as f32, 1.36 as f32],
            vec![4.41 as f32, -2.16 as f32],
        ],
        &device,
    )?;

    let attn = self_attn.attention(&encoding_matrix)?;
    println!("{}", attn);
    Ok(())
}
