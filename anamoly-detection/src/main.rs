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

fn z_score_normalize(data: &Tensor) -> Result<Tensor> {
    let mean = data.mean(0)?;
    let squared_diff = data.broadcast_sub(&mean)?.sqr()?;
    let variance = squared_diff.mean(0)?;
    let std_dev = variance.sqrt()?;
    let normalized = data.broadcast_sub(&mean)?.broadcast_div(&std_dev)?;
    Ok(normalized)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Data CSV file from https://www.kaggle.com/c/eecs498/data
    #[arg(long)]
    data_csv: String,

    #[arg(long, short, default_value = "false")]
    print: bool,

    #[arg(long, default_value = "0.001")]
    episilon: f64,
}

fn p_x(
    x: &Tensor,
    mean: &Tensor,
    two_variance: &Tensor,
    two_pi_sqrt_std_dev: &Tensor,
) -> Result<f64> {
    let px = x
        .broadcast_sub(mean)?
        .sqr()?
        .broadcast_div(two_variance)?
        .exp()?
        .broadcast_mul(two_pi_sqrt_std_dev)?
        .recip()?;
    let px = px.to_vec1::<f64>()?.into_iter().fold(1.0, |acc, x| acc * x);
    Ok(px)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = Device::cuda_if_available(0)?;
    let data = load_dataset(&args.data_csv, &device)?;

    let data = z_score_normalize(&data)?;

    let mean = data.mean(0)?;
    let variance = data.broadcast_sub(&mean)?.sqr()?.mean(0)?;
    let std_dev = variance.sqrt()?;

    let two_variance = variance.broadcast_mul(&Tensor::new(2.0, &device)?)?;
    let two_pi_sqrt_std_dev =
        std_dev.broadcast_mul(&Tensor::new(2.0 * std::f64::consts::PI, &device)?.sqrt()?)?;

    let rows = data.shape().dims2()?.0;
    let mut anamolies = 0;
    for row in 0..rows {
        let row_tensor = data
            .index_select(&Tensor::new(&[row as u32], &device)?, 0)?
            .squeeze(0)?;
        let px = p_x(&row_tensor, &mean, &two_variance, &two_pi_sqrt_std_dev)?;
        if px < args.episilon {
            anamolies += 1;
            if args.print {
                println!("Anamoly: {}", row + 1);
            }
        }
    }

    println!("Anamolies: {}, Total: {}", anamolies, rows);

    Ok(())
}
