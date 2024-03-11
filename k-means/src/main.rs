extern crate csv;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use clap::Parser;
use rand::prelude::*;

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(1)?;
    let x2 = x2.unsqueeze(2)?;
    Ok(x1.broadcast_sub(&x2)?.sqr()?.sum(D::Minus1)?.sqrt()?)
}

fn k_means(x: &Tensor, k: usize, max_iter: i64, device: &Device) -> Result<(Tensor, Tensor)> {
    let (n, d) = x.dims2()?;
    let mut rng = rand::thread_rng();
    let indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(rng);
    let centroid_idx = indices[..k]
        .to_vec()
        .into_iter()
        .map(|x| x as i64)
        .collect::<Vec<_>>();
    let centroid_idx_tensor = Tensor::from_slice(&centroid_idx.as_slice(), (k,), device)?;
    let mut centers = x.index_select(&centroid_idx_tensor, 2)?;

    for _ in 0..max_iter {
        let dist = cdist(x, &centers)?;
        let cluster_assignments = dist.argmin(D::Minus1)?;
        let mut new_centers = Tensor::zeros_like(&centers)?;
        let mut counts = Tensor::zeros(k, D::Int64)?;
        for i in 0..n {
            let label = labels.get(i)?;
            new_centers.index_add_1(label, x.get(i)?)?;
            counts.index_add_1(label, &Tensor::ones(1, D::Int64)?)?;
        }
        for i in 0..k {
            if counts.get(i)? > 0 {
                new_centers.index_div_1(i, counts.get(i)?)?;
            }
        }
        if (new_centers - centers).abs().max(D::Minus1)?.get(0)? < 1e-6 {
            break;
        }
        centers = new_centers;
    }
    let dist = cdist(x, &centers)?;
    let (_, labels) = dist.min(D::Minus1, true)?;
    Ok((centers, labels))
}

fn main() {
    println!("Hello, world!");
}
