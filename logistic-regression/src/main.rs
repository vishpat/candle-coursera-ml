extern crate csv;
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use clap::Parser;
use rand::prelude::*;
use std::iter;
use std::rc::Rc;

// Implement Logistic Regression model using Gradient Descent
// https://www.youtube.com/watch?v=4u81xU7BIOc
struct LogisticRegression {
    thetas: Tensor,
    device: Rc<Device>,
}

fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    Ok((xs.neg()?.exp()? + 1.0)?.recip()?)
}

impl LogisticRegression {
    fn new(feature_cnt: usize, device: Rc<Device>) -> Result<Self> {
        let thetas: Vec<f32> = vec![0.0; feature_cnt];
        let thetas = Tensor::from_vec(thetas, (feature_cnt,), &device)?;
        Ok(Self { thetas, device })
    }

    fn hypothesis(&self, x: &Tensor) -> Result<Tensor> {
        Ok(sigmoid(&x.matmul(&self.thetas.unsqueeze(1)?)?.squeeze(1)?)?)
    }

    fn loss(&self, y1: &Tensor, y2: &Tensor) -> Result<f32> {
        let diff = y1.sub(y2)?;
        let loss = diff.mul(&diff)?.mean(D::Minus1)?.to_scalar::<f32>()?;
        Ok(loss)
    }

    fn cost(&self, x: &Tensor, y: &Tensor) -> Result<f32> {
        let device = Device::cuda_if_available(0)?;
        let (m, _) = x.shape().dims2()?;
        let h = self.hypothesis(x)?;
        let log_h = h.log()?;
        let one_array = Tensor::from_iter(iter::repeat(1.0f32).take(m), &device)?;
        let log_1_minus_h = one_array.sub(&h)?.log()?;

        let one_array = Tensor::from_iter(iter::repeat(1.0f32).take(m), &device)?;
        let one_minus_y = one_array.sub(y)?;
        let cost = y
            .mul(&log_h)?
            .add(&one_minus_y.mul(&log_1_minus_h)?)?
            .broadcast_div(&Tensor::new(-1.0 * m as f32, &device)?)?
            .sum(D::Minus1)?
            .to_scalar()?;
        Ok(cost)
    }

    fn train(&mut self, x: &Tensor, y: &Tensor, learning_rate: f32) -> Result<()> {
        let m = y.shape().dims1()?;
        let predictions = self.hypothesis(x)?;
        let deltas = predictions.sub(y)?;
        let gradient = x
            .t()?
            .matmul(&deltas.unsqueeze(D::Minus1)?)?
            .broadcast_div(&Tensor::new(m as f32, &self.device)?)?;
        let gradient = gradient.squeeze(D::Minus1)?.squeeze(D::Minus1)?;
        self.thetas = self
            .thetas
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
    let (m, n) = dataset.train_images.shape().dims2()?;
    let training_images = dataset.train_images;
    let training_images_vec = training_images
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let training_images = Tensor::from_vec(training_images_vec, (m, n), &device)?;
    let training_labels = dataset.train_labels;
    let training_labels_vec = training_labels
        .to_vec1::<u8>()?
        .into_iter()
        .map(|x| if x == args.digit { 1.0 } else { 0.0 })
        .collect::<Vec<f32>>();
    let len = training_labels_vec.len();
    let training_labels = Tensor::from_vec(training_labels_vec, (len,), &device)?;

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
            model.train(&train_data, &train_labels, args.learning_rate)?;
            let predictions = model.hypothesis(&train_data)?;
            let loss = model.loss(&predictions, &train_labels)?;
            sum_loss += loss;
        }
        if args.progress && epoch % 1000 == 0 {
            let cost = model.cost(&training_images, &training_labels)?;
            println!(
                "epoch: {epoch}, cost: {cost},  loss: {}",
                sum_loss / n_batches as f32
            );
        }
    }

    let test_images = dataset.test_images;
    let (m, n) = test_images.shape().dims2()?;
    let test_images_vec = test_images
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let test_images = Tensor::from_vec(test_images_vec, (m, n), &device)?;
    let test_labels = dataset.test_labels;
    let test_labels_vec = test_labels
        .to_vec1::<u8>()?
        .into_iter()
        .map(|x| if x == args.digit { 1f32 } else { 0f32 })
        .collect::<Vec<f32>>();
    let len = test_labels_vec.len();
    let test_labels = Tensor::from_vec(test_labels_vec, (len,), &device)?;
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
    println!("Accuracy: {}", accuracy);

    Ok(())
}
