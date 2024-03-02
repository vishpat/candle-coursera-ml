use anyhow::Result;
use candle::{Device, Tensor, D};
use candle_nn::{loss, ops, Conv2d, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use rand::prelude::*;
use std::rc::Rc;

const IMAGE_DIM: usize = 28 * 28;
const LABELS: usize = 10;

struct Dataset {
    pub training_data: Tensor,
    pub training_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
    pub feature_cnt: usize,
}

fn load_tensors(csv: &str, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    let mut rdr = csv::Reader::from_path(csv)?;
    for result in rdr.records() {
        let record = result?;
        let label = record.get(0).unwrap().parse::<f32>()?;
        let mut features = Vec::new();
        for i in 1..record.len() {
            features.push(record.get(i).unwrap().parse::<f32>()?);
        }
        labels.push(label);
        data.push(features);
    }

    let data = data.into_iter().flatten().collect::<Vec<f32>>();
    let data = Tensor::from_slice(&data, (labels.len(), IMAGE_DIM), device)?;
    let labels = Tensor::from_slice(&labels, (labels.len(),), device)?;

    Ok((data, labels))
}

fn load_dataset(train_csv: &str, test_csv: &str, device: &Device) -> Result<Dataset> {
    let (training_data, training_labels) = load_tensors(train_csv, device)?;
    let (test_data, test_labels) = load_tensors(test_csv, device)?;

    Ok(Dataset {
        training_data,
        training_labels,
        test_data,
        test_labels,
        feature_cnt: IMAGE_DIM,
    })
}
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    train_csv: String,

    #[arg(long)]
    test_csv: String,

    // Print the Cost and Loss at each epoch
    #[arg(long, default_value_t = false)]
    progress: bool,

    // The learning rate
    #[arg(long, default_value = "0.01")]
    learning_rate: f32,

    // The regularization parameter
    #[arg(long, default_value = "0.01")]
    regularization: f32,

    // The number of epochs
    #[arg(long, default_value = "10000")]
    epochs: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Rc::new(Device::cuda_if_available(0)?);
    let dataset = load_dataset(&args.train_csv, &args.test_csv, &device)?;

    Ok(())
}
