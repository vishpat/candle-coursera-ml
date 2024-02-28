extern crate csv;
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use clap::Parser;
use core::panic;
use rand::prelude::*;
use std::fs::File;
use std::rc::Rc;

struct Dataset {
    pub training_data: Tensor,
    pub training_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
    pub feature_cnt: usize,
}

// Implement Linear Regression model using Gradient Descent
// https://www.youtube.com/watch?v=UVCFaaEBnTE
struct LinearRegression {
    thetas: Tensor,
    device: Rc<Device>,
}

impl LinearRegression {
    fn new(feature_cnt: usize, device: Rc<Device>) -> Result<Self> {
        let thetas: Vec<f32> = vec![0.0; feature_cnt];
        let thetas = Tensor::from_vec(thetas, (feature_cnt,), &device)?;
        Ok(Self { thetas, device })
    }

    fn hypothesis(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.thetas.unsqueeze(1)?)?.squeeze(1)?)
    }

    fn cost(&self, x: &Tensor, y: &Tensor) -> Result<f32> {
        let m = y.shape().dims1()?;
        let predictions = self.hypothesis(x)?;
        let deltas = predictions.sub(y)?;
        let cost = deltas
            .mul(&deltas)?
            .mean(D::Minus1)?
            .div(&Tensor::new(2.0 * m as f32, &self.device)?)?
            .to_scalar::<f32>()?;
        Ok(cost)
    }

    fn loss(&self, y1: &Tensor, y2: &Tensor) -> Result<f32> {
        let diff = y1.sub(y2)?;
        let loss = diff.mul(&diff)?.mean(D::Minus1)?.to_scalar::<f32>()?;
        Ok(loss)
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

fn r2_score(predictions: &Tensor, labels: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let mean = labels.mean(D::Minus1)?;

    let ssr = labels.sub(predictions)?;
    let ssr = ssr.mul(&ssr)?.sum(D::Minus1)?;

    let sst = labels.broadcast_sub(&mean)?;
    let sst = sst.mul(&sst)?.sum(D::Minus1)?;

    let tmp = ssr.div(&sst)?.to_scalar::<f32>()?;

    Ok(1.0 - tmp)
}

const LEARNING_RATE: f32 = 0.01;
const EPOCHS: i32 = 10000;
const BATCH_SIZE: usize = 100;

fn insurance_dataset(file_path: &str, device: &Device) -> Result<Dataset> {
    // https://www.kaggle.com/mirichoi0218/insurance

    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut data: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<f32> = vec![];

    const FEATURE_CNT: usize = 7;
    const MALE: f32 = 0.5;
    const FEMALE: f32 = -0.5;

    const YES: f32 = 0.5;
    const NO: f32 = -0.5;

    const NORTHWEST: f32 = 0.25;
    const NORTHEAST: f32 = -0.25;
    const SOUTHWEST: f32 = 0.5;
    const SOUTHEAST: f32 = -0.5;

    for result in rdr.records() {
        let record = result?;
        let age: f32 = (record[0].parse::<u32>()? as f32) / 100.0;
        let gender = match record[1].parse::<String>()?.as_str() {
            "male" => MALE,
            "female" => FEMALE,
            _ => panic!("Invalid Gender"),
        };
        let bmi: f32 = record[2].parse::<f32>()? / 100.0;
        let children: f32 = record[3].parse()?;
        let smoker = match record[4].parse::<String>()?.as_str() {
            "yes" => YES,
            "no" => NO,
            _ => panic!("Invalid Smoker"),
        };
        let region = match record[5].parse::<String>()?.as_str() {
            "northwest" => NORTHWEST,
            "northeast" => NORTHEAST,
            "southwest" => SOUTHWEST,
            "southeast" => SOUTHEAST,
            _ => panic!("Invalid Region"),
        };
        let charges: f32 = record[6].parse()?;

        // IMPORTANT: The first column of the row needs to be 1.0 for the bias term
        let row = vec![1.0, age, gender, bmi, children, smoker, region];
        data.push(row);

        let label = charges;
        labels.push(label);
    }
    let training_size = labels.len() * 8 / 10;
    let training_data = data[..training_size].to_vec();
    let training_labels = labels[..training_size].to_vec();

    let training_data = training_data
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<f32>>();
    let training_data_tensor =
        Tensor::from_slice(&training_data, (training_labels.len(), FEATURE_CNT), device)?;
    let training_labels_tensor =
        Tensor::from_slice(&training_labels, (training_labels.len(),), device)?;

    let test_data = data[training_size..].to_vec();
    let test_labels = labels[training_size..].to_vec();

    let test_data = test_data.iter().flatten().copied().collect::<Vec<f32>>();
    let test_data_tensor =
        Tensor::from_slice(&test_data, (test_labels.len(), FEATURE_CNT), device)?;
    let test_labels_tensor = Tensor::from_slice(&test_labels, (test_labels.len(),), device)?;

    Ok(Dataset {
        training_data: training_data_tensor,
        training_labels: training_labels_tensor,
        test_data: test_data_tensor,
        test_labels: test_labels_tensor,
        feature_cnt: FEATURE_CNT,
    })
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    data_csv: String,

    // Print the Cost and Loss at each epoch
    #[arg(long, default_value_t = false)]
    progress: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let file_path = args.data_csv;

    let device = Rc::new(Device::cuda_if_available(0)?);

    let dataset = insurance_dataset(&file_path, &device)?;

    let mut model = LinearRegression::new(dataset.feature_cnt, device)?;
    let (training_size, _) = dataset.training_data.shape().dims2()?;
    let n_batches = training_size / BATCH_SIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 0..EPOCHS {
        let mut sum_loss = 0.0;
        batch_idxs.shuffle(&mut rand::thread_rng());
        for batch_idx in batch_idxs.iter() {
            let train_data = dataset
                .training_data
                .narrow(0, batch_idx * BATCH_SIZE, BATCH_SIZE)?;
            let train_labels =
                dataset
                    .training_labels
                    .narrow(0, batch_idx * BATCH_SIZE, BATCH_SIZE)?;
            model.train(&train_data, &train_labels, LEARNING_RATE)?;
            let predictions = model.hypothesis(&train_data)?;
            let loss = model.loss(&predictions, &train_labels)?;
            sum_loss += loss;
        }
        if args.progress && epoch % 1000 == 0 {
            let cost = model.cost(&dataset.training_data, &dataset.training_labels)?;
            println!(
                "epoch: {epoch}, cost: {cost},  loss: {}",
                sum_loss / n_batches as f32
            );
        }
    }

    let predictions = model.hypothesis(&dataset.test_data)?;
    let r2 = r2_score(&predictions, &dataset.test_labels).unwrap();
    println!("r2: {r2}");

    Ok(())
}
