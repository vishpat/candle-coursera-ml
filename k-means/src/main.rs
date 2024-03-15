extern crate csv;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use clap::Parser;
use rand::prelude::*;

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(1)?;
    Ok(x1.broadcast_sub(&x2)?.sqr()?.sum(D::Minus1)?.sqrt()?.transpose(D::Minus1, D::Minus2)?)
}

fn load_dataset(file_path: &str, device: &Device) -> Result<Tensor> {
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let mut row = vec![];
        for i in 1..5 {
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

fn k_means(data: &Tensor, k: usize, max_iter: i64, device: &Device) -> Result<(Tensor, Tensor)> {
    let (n, _) = data.dims2()?;
    let mut rng = rand::thread_rng();
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(&mut rng);

    let centroid_idx = indices[..k]
        .iter()
        .copied()
        .map(|x| x as i64)
        .collect::<Vec<_>>();
    
    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros((n,), DType::U32, device)?;
    for _ in 0..max_iter {
        let dist = cdist(data, &centers)?;
        cluster_assignments = dist.argmin(D::Minus1)?;
        println!("Cluster Assignments {cluster_assignments}");
        centers = Tensor::zeros_like(&centers)?;
    }
    Ok((centers, cluster_assignments))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Data CSV file from https://www.kaggle.com/datasets/uciml/iris/data
    #[arg(long)]
    data_csv: String,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    let data = load_dataset(&args.data_csv, &device).unwrap();
    let (centers, cluster_assignments) = k_means(&data, 3, 1, &device)?;
    Ok(())
}
