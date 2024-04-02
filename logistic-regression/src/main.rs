extern crate csv;
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use clap::Parser;
use rand::prelude::*;
use std::rc::Rc;

// Implement Logistic Regression model using Gradient Descent
// https://www.youtube.com/watch?v=4u81xU7BIOc
struct LogisticRegression {
    weights: Tensor,
    bias: Tensor,
    device: Rc<Device>,
}

fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    Ok((xs.neg()?.exp()? + 1.0)?.recip()?)
}

impl LogisticRegression {
    fn new(feature_cnt: usize, device: Rc<Device>) -> Result<Self> {
        let weights: Vec<f32> = vec![0.0; feature_cnt];
        let weights = Tensor::from_vec(weights, (feature_cnt,), &device)?;
        let bias = Tensor::new(0.0f32, &device)?;
        Ok(Self {
            weights,
            bias,
            device,
        })
    }

    fn hypothesis(&self, x: &Tensor) -> Result<Tensor> {
        Ok(sigmoid(
            &x.matmul(&self.weights.unsqueeze(1)?)?
                .squeeze(1)?
                .broadcast_add(&self.bias)?,
        )?)
    }

    fn loss(&self, y1: &Tensor, y2: &Tensor) -> Result<f32> {
        let diff = y1.sub(y2)?;
        let loss = diff.mul(&diff)?.mean(D::Minus1)?.to_scalar::<f32>()?;
        Ok(loss)
    }

    fn train(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        learning_rate: f32,
        regularization: f32,
    ) -> Result<()> {
        let m = y.shape().dims1()?;
        let predictions = self.hypothesis(x)?;
        let deltas = predictions.sub(y)?;
        let regularization = self
            .weights
            .broadcast_mul(&Tensor::new(regularization / m as f32, &self.device)?)?;

        let gradient = x
            .t()?
            .matmul(&deltas.unsqueeze(D::Minus1)?)?
            .broadcast_div(&Tensor::new(m as f32, &self.device)?)?;
        let gradient = gradient
            .squeeze(D::Minus1)?
            .squeeze(D::Minus1)?
            .add(&regularization)?;

        self.weights = self
            .weights
            .sub(&gradient.broadcast_mul(&Tensor::new(learning_rate, &self.device)?)?)?;
        let gradient = deltas.mean(D::Minus1)?;
        self.bias = self
            .bias
            .sub(&gradient.broadcast_mul(&Tensor::new(learning_rate, &self.device)?)?)?;
        Ok(())
    }
}

const BATCH_SIZE: usize = 100;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Print the Cost and Loss at each epoch
    #[arg(long, default_value_t = false)]
    progress: bool,
    // The learning rate
    #[arg(long, default_value = "0.01")]
    learning_rate: f32,

    // Regularization parameter
    #[arg(long, default_value = "0.1")]
    regularization: f32,

    // The number of epochs
    #[arg(long, default_value = "10000")]
    epochs: i32,

    // The digit to classify
    #[arg(long, default_value = "0")]
    digit: u8,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Rc::new(Device::cuda_if_available(0)?);

    let dataset = candle_datasets::vision::mnist::load()?;
    let (_, n) = dataset.train_images.shape().dims2()?;
    let training_images = dataset.train_images.to_device(&device)?;
    let training_labels = dataset.train_labels.to_device(&device)?;
    let training_labels_vec = training_labels
        .to_vec1::<u8>()?
        .into_iter()
        .map(|x| if x == args.digit { 1.0 } else { 0.0 })
        .collect::<Vec<f32>>();
    let len = training_labels_vec.len();
    let training_labels = Tensor::from_vec(training_labels_vec, (len,), &device)?;

    let test_images = dataset.test_images.to_device(&device)?;
    let test_labels = dataset.test_labels.to_device(&device)?;
    let test_labels_vec = test_labels
        .to_vec1::<u8>()?
        .into_iter()
        .map(|x| if x == args.digit { 1f32 } else { 0f32 })
        .collect::<Vec<f32>>();
    let len = test_labels_vec.len();
    let test_labels = Tensor::from_vec(test_labels_vec, (len,), &device)?;

    let mut model = LogisticRegression::new(n, device.clone())?;
    let (training_size, _) = training_images.shape().dims2()?;
    let n_batches = training_size / BATCH_SIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 0..args.epochs {
        let mut sum_loss = 0.0;
        batch_idxs.shuffle(&mut rand::thread_rng());
        for batch_idx in batch_idxs.iter() {
            let train_data = training_images.narrow(0, batch_idx * BATCH_SIZE, BATCH_SIZE)?;
            let train_labels = training_labels.narrow(0, batch_idx * BATCH_SIZE, BATCH_SIZE)?;
            model.train(
                &train_data,
                &train_labels,
                args.learning_rate,
                args.regularization,
            )?;
            let predictions = model.hypothesis(&train_data)?;
            let loss = model.loss(&predictions, &train_labels)?;
            sum_loss += loss;
        }
        if args.progress && epoch % 1000 == 0 {
            let predictions = model.hypothesis(&test_images)?;
            let predictions_vec = predictions
                .to_vec1::<f32>()?
                .into_iter()
                .map(|x| if x > 0.5 { 1f32 } else { 0f32 })
                .collect::<Vec<f32>>();
            let predictions = Tensor::from_vec(predictions_vec, (len,), &device)?;

            let accuracy = predictions
                .eq(&test_labels)?
                .to_vec1::<u8>()?
                .into_iter()
                .map(f32::from)
                .sum::<f32>()
                / len as f32;
            println!(
                "epoch: {epoch}, loss: {}, Test Accuracy: {}",
                sum_loss / n_batches as f32,
                accuracy
            );
        }
    }

    Ok(())
}
