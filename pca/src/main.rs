use anyhow::{Ok, Result};
use candle_core::{Device, Tensor, D};
use clap::Parser;
use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;

fn load_dataset(file_path: &str, device: &Device) -> Result<Tensor> {
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let mut row = Vec::new();
        for i in 2..32 {
            let value = record[i].parse::<f32>()?;
            row.push(value);
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

fn cov(data: &Tensor, device: &Device) -> Result<Tensor> {
    let mean = data.mean(0)?;
    let centered = data.broadcast_sub(&mean)?;
    let (m, _) = data.shape().dims2()?;
    let cov = centered
        .transpose(D::Minus1, D::Minus2)?
        .matmul(&centered)?
        .broadcast_div(&Tensor::new(m as f32, device)?)?;

    Ok(cov)
}

fn pca(normalized_data: &Tensor, device: &Device, variance: f32) -> Result<Tensor> {
    let (_, n) = normalized_data.shape().dims2()?;
    let cov = cov(normalized_data, device)?;
    let vec: Vec<f32> = cov
        .to_device(&Device::Cpu)?
        .to_vec2()?
        .into_iter()
        .flatten()
        .collect();
    let dmatrix = DMatrix::from_vec(n, n, vec);
    let eig = SymmetricEigen::new(dmatrix);
    let eigen_values = eig.eigenvalues.data.as_vec();
    let total = eigen_values.iter().sum::<f32>();
    let mut k = 0;
    for i in 0..n {
        let var = eigen_values[0..i].iter().sum::<f32>() / total;
        if var > variance {
            println!("{} components explain {}% of the variance", i, var * 100.0);
            k = i;
            break;
        }
    }

    let eigen_vectors = eig.eigenvectors.data.as_vec();
    let eigen_vectors = eigen_vectors
        .chunks(n)
        .take(k)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let eigen_vectors = Tensor::from_slice(eigen_vectors.as_slice(), (k, n), device)?;
    Ok(eigen_vectors)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Data CSV file from https://www.kaggle.com/datasets/uciml/iris/data
    #[arg(long)]
    data_csv: String,

    #[arg(long, default_value = "0.95")]
    variance: f32,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    let data = load_dataset(&args.data_csv, &device).unwrap();
    let normalized_data = z_score_normalize(&data)?;
    let reduce = pca(&normalized_data, &device, args.variance)?;
    let compressed_data = data.matmul(&reduce.transpose(D::Minus1, D::Minus2)?)?;
    println!("Compressed data {:?}", compressed_data);
    Ok(())
}
