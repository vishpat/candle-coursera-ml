use anyhow::Ok;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use rand::prelude::*;
use rand::Rng;
use std::f64::consts::PI;
use std::rc::Rc;

const INPUT_COUNT: usize = 3;
const LAYER1_COUNT: usize = 50;
const LAYER2_COUNT: usize = 50;
const OUTPUT_COUNT: usize = 1;

const BATCH_SIZE: usize = 1000;

struct Dataset {
    pub training_data: Tensor,
    pub training_values: Tensor,
    pub test_data: Tensor,
    pub test_values: Tensor,
}

fn func(x1: f32, x2: f32, x3: f32) -> f32 {
    2.0 * (x1.abs() + 1.0).ln() + (3.0 * x2).sin() * (-0.5 * x3).exp() + 0.5 * x1 * x3
}

fn generate_nonlinear_data(n_samples: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_values = Vec::with_capacity(n_samples);
    let mut y_values = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x1 = rng.gen_range(-5.0..=5.0) as f32;
        let x2 = rng.gen_range(-PI..=PI) as f32;
        let x3 = rng.gen_range(-3.0..=3.0) as f32;
        let noise = rng.gen_range(-0.5..=0.5) as f32;
        let y = (func(x1, x2, x3) + noise) as f32;
        x_values.push(vec![x1, x2, x3]);
        y_values.push(y);
    }

    (x_values, y_values)
}

fn load_tensors(samples: u32, device: &Device) -> Result<(Tensor, Tensor)> {
    let (x_values, y_values) = generate_nonlinear_data(samples as usize);

    let x_values = x_values.into_iter().flatten().collect::<Vec<f32>>();
    let x_values = x_values.as_slice();
    let x_tensor = Tensor::from_slice(x_values, &[samples as usize, 3], device)?;

    let y_values = y_values.into_iter().collect::<Vec<f32>>();
    let y_values = y_values.as_slice();
    let y_tensor = Tensor::from_slice(y_values, &[samples as usize], device)?;

    Ok((x_tensor, y_tensor))
}

fn load_dataset(device: &Device) -> Result<Dataset> {
    let (training_data, training_values) = load_tensors(5000, device)?;
    let (test_data, test_values) = load_tensors(2000, device)?;

    Ok(Dataset {
        training_data,
        training_values,
        test_data,
        test_values,
    })
}

fn r_square(labels: &Tensor, predictions: &Tensor) -> Result<f32> {
    let mean = labels.mean(D::Minus1)?;

    let ssr = labels.sub(predictions)?;
    let ssr = ssr.mul(&ssr)?.sum(D::Minus1)?;

    let sst = labels.broadcast_sub(&mean)?;
    let sst = sst.mul(&sst)?.sum(D::Minus1)?;

    let tmp = ssr.div(&sst)?.to_scalar::<f32>()?;

    Ok(1.0 - tmp)
}

struct Mlp {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(INPUT_COUNT, LAYER1_COUNT, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_COUNT, LAYER2_COUNT, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_COUNT, OUTPUT_COUNT, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.tanh()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.tanh()?;
        Ok(self.ln3.forward(&xs)?)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Print the Cost and Loss at each epoch
    #[arg(long, default_value_t = false)]
    progress: bool,

    // The learning rate
    #[arg(long, default_value = "0.01")]
    learning_rate: f64,

    // The regularization parameter
    #[arg(long, default_value = "0.01")]
    regularization: f32,

    // The number of epochs
    #[arg(long, default_value = "5000")]
    epochs: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Rc::new(Device::cuda_if_available(0)?);
    let dataset = load_dataset(&device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Mlp::new(vs)?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;

    let test_data = dataset.test_data.to_device(&device)?;
    let test_values = dataset
        .test_values
        .to_dtype(DType::F32)?
        .to_device(&device)?;

    let (training_size, _) = dataset.training_data.shape().dims2()?;
    let n_batches = training_size / BATCH_SIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 1..args.epochs {
        batch_idxs.shuffle(&mut rand::thread_rng());
        for batch_idx in batch_idxs.iter() {
            let train_data = dataset
                .training_data
                .narrow(0, batch_idx * BATCH_SIZE, BATCH_SIZE)?;
            let train_values =
                dataset
                    .training_values
                    .narrow(0, batch_idx * BATCH_SIZE, BATCH_SIZE)?;
            let logits = model.forward(&train_data)?;
            let loss = loss::mse(&logits.squeeze(1)?, &train_values)?;
            sgd.backward_step(&loss)?;
        }
        let test_logits = model.forward(&test_data)?;
        if args.progress && epoch % 100 == 0 {
            println!(
                "{epoch:4} test r2: {:?}",
                r_square(&test_values, &test_logits.squeeze(1)?)?
            );
        }
    }

    Ok(())
}
