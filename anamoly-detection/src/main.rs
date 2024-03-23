extern crate csv;
use std::vec;

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;

fn load_dataset(file_path: &str, device: &Device) -> Result<Tensor> {
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let mut row = vec![];
        for i in 1..4 {
            row.push(record[i].parse::<f64>()?);
        }
        data.push(row);
    }
    let feature_cnt = data[0].len();
    let sample_cnt = data.len();
    let data = data.into_iter().flatten().collect::<Vec<_>>();
    let data = Tensor::from_slice(data.as_slice(), (sample_cnt, feature_cnt), device)?;
    Ok(data)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Data CSV file from https://www.kaggle.com/c/eecs498/data
    #[arg(long)]
    data_csv: String,
}

fn p(
    x: &Tensor,
    mean: &Tensor,
    two_variance: &Tensor,
    two_pi_sqrt_std_dev: &Tensor,
    device: &Device,
) -> Result<f64> {
    let px = x
        .broadcast_sub(&mean)?
        .div(&two_variance)?
        .broadcast_mul(&Tensor::new(-1.0, &device)?)?
        .exp()?
        .broadcast_div(&two_pi_sqrt_std_dev)?;
    let px = px.to_vec1::<f64>()?.into_iter().fold(1.0, |acc, x| acc * x);
    Ok(px)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    let data = load_dataset(&args.data_csv, &device)?;

    let mean = data.mean(0)?;
    let variance = data.broadcast_sub(&mean)?.sqr()?.mean(0)?;
    let std_dev = variance.sqrt()?;
    let two_variance = variance.broadcast_mul(&Tensor::new(2.0, &device)?)?;
    let two_pi_sqrt_std_dev =
        std_dev.broadcast_mul(&Tensor::new(2.0 * std::f64::consts::PI, &device)?.sqrt()?)?;

    Ok(())
}
