extern crate csv;
use std::vec;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use clap::Parser;
use rand::prelude::*;

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(1)?;
    Ok(x1
        .broadcast_sub(&x2)?
        .sqr()?
        .sum(D::Minus1)?
        .sqrt()?
        .transpose(D::Minus1, D::Minus2)?)
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
        let mut centers_vec = vec![];
        for i in 0..k {
            let mut indices = vec![];
            cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .for_each(|(j, x)| {
                    if *x == i as u32 {
                        indices.push(j as u32);
                    }
                });
            let indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
            let cluster_data = data.index_select(&indices, 0)?;
            let mean = cluster_data.mean(0)?;
            centers_vec.push(mean);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    Ok((centers, cluster_assignments))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Data CSV file from https://www.kaggle.com/datasets/uciml/iris/data
    #[arg(long)]
    data_csv: String,

    // Number of clusters
    #[arg(long, default_value = "3")]
    k: usize,

    // Maximum number of iterations
    #[arg(long, default_value = "100")]
    max_iter: i64,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    let data = load_dataset(&args.data_csv, &device).unwrap();
    let (centers, cluster_assignments) = k_means(&data, args.k, args.max_iter, &device)?;
    println!("{}", centers);
    println!("{}", cluster_assignments);
    let cluster_sizes = cluster_assignments.to_vec1::<u32>()?;
    for i in 0..args.k {
        let size = cluster_sizes.iter().filter(|&&x| x == i as u32).count();
        println!("Cluster {} size: {}", i, size);
    }
    Ok(())
}
