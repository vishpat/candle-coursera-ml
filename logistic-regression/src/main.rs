extern crate csv;
use anyhow::Result;
use candle_core::{Device, Tensor, D};
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

    #[allow(unused)]
    fn cost(&self, x: &Tensor, y: &Tensor) -> Result<f32> {
        let device = Device::cuda_if_available(0)?;
        let (m, n) = x.shape().dims2()?;
        let H = self.hypothesis(x)?;
        let log_H = H.log()?;
        let one_array = Tensor::from_iter(iter::repeat(1.0f32).take(m), &device)?;
        let log_1_minus_H = one_array.sub(&H)?.log()?;

        let one_array = Tensor::from_iter(iter::repeat(1.0f32).take(m), &device)?;
        let one_minus_y = one_array.sub(y)?;
        let cost = y
            .mul(&log_H)?
            .add(&one_minus_y.mul(&log_1_minus_H)?)?
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

const LEARNING_RATE: f32 = 0.01;
const ITERATIONS: i32 = 10000;

fn main() -> Result<()> {
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
        .map(|x| if x == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<f32>>();
    let len = training_labels_vec.len();
    let training_labels = Tensor::from_vec(training_labels_vec, (len,), &device)?;

    let mut model = LogisticRegression::new(n, device.clone())?;

    let mut cost: f32 = 0.0;
    for i in 0..ITERATIONS {
        cost = model.cost(&training_images, &training_labels)?;

        if i % 1000 == 0 {
            println!("Cost: {}", cost);
        }

        model.train(&training_images, &training_labels, LEARNING_RATE)?;
    }
    println!("Cost: {}", cost);

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
        .map(|x| if x == 0 { 1f32 } else { 0f32 })
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
