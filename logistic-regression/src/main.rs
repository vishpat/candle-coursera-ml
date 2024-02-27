extern crate csv;
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use clap::Parser;
use core::panic;
use std::char::from_digit;
use std::fs::File;
use std::{array, iter};
use std::rc::Rc;
use polars::prelude::*;

struct Dataset {
    pub training_data: Tensor,
    pub training_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
    pub feature_cnt: usize,
}

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
    fn cost(&self, x: &Tensor, y: &Tensor, device: Rc<Device>) -> Result<Tensor> {
        let (m, n) = x.shape().dims2()?;
        let H = self.hypothesis(x)?;
        let log_H = H.log()?;
        let one_array = Tensor::from_iter(iter::repeat(1.0f32).take(n), &device)?;
        let log_1_minus_H = one_array.sub(&H)?.log()?;

        let one_array = Tensor::from_iter(iter::repeat(1.0f32).take(m), &device)?;
        let one_minus_y = one_array.sub(y)?;
        let cost = y
            .mul(&log_H)?
            .add(&one_minus_y.mul(&log_1_minus_H)?)?
            .broadcast_div(&Tensor::new(-1.0 * m as f32, &device)?)?;
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
const ITERATIONS: i32 = 100000;

fn income_dataset(
    training_file_path: &str,
    test_file_path: &str,
    device: &Device,
) -> Result<Dataset> {
    // https://www.kaggle.com/datasets/nimapourmoradi/adult-incometrain-test-dataset/data

    let dataset = CsvReader::from_path(training_file_path)?
        .has_header(true)
        .finish()?;

    println!("{:?}", dataset);


    let dummy1 = Tensor::from_slice(&[1.0, 2.0], (1,), device)?;
    let dummy2 = Tensor::from_slice(&[1.0, 2.0], (1,), device)?;
    let dummy3 = Tensor::from_slice(&[1.0, 2.0], (1,), device)?;
    let dummy4 = Tensor::from_slice(&[1.0, 2.0], (1,), device)?;

    Ok(Dataset {
        training_data: dummy1,
        training_labels: dummy2,
        test_data: dummy3,
        test_labels: dummy4,
        feature_cnt: 0,
    })
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    training_csv: String,

    #[arg(long)]
    test_csv: String,
}
fn main() -> Result<()> {
    let args = Args::parse();

    let training_file_path = args.training_csv;
    let test_file_path = args.test_csv;

    let device = Rc::new(Device::cuda_if_available(0)?);

    let dataset = income_dataset(&training_file_path, &test_file_path, &device)?;

//    let mut model = LogisticRegression::new(dataset.feature_cnt, device)?;
//
//    for _ in 0..ITERATIONS {
//        model.train(
//            &dataset.training_data,
//            &dataset.training_labels,
//            LEARNING_RATE,
//        )?;
//    }
//
//    let predictions = model.hypothesis(&dataset.test_data)?;
//
    Ok(())
}
