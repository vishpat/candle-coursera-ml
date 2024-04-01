extern crate csv;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::vec;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
struct Rating {
    user: u32,
    movie: u32,
    rating_u32: u32,
}

impl Rating {
    fn rating(&self) -> f32 {
        self.rating_u32 as f32 / 10.0
    }
}

// Step 2: Implement `PartialOrd` and `Ord` for the struct.
impl PartialOrd for Rating {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rating {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by `userId`, then by `movieId`.
        self.user
            .cmp(&other.user)
            .then_with(|| self.movie.cmp(&other.movie))
    }
}

fn load_ratings(file_path: &str) -> Result<(HashSet<u32>, HashSet<u32>, HashSet<Rating>)> {
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut users: HashSet<u32> = HashSet::new();
    let mut movies: HashSet<u32> = HashSet::new();
    let mut ratings: HashSet<Rating> = HashSet::new();

    for result in rdr.records() {
        let record = result?;
        let user: u32 = record[0].parse()?;
        let movie: u32 = record[1].parse()?;
        let rating: f32 = record[2].parse()?;
        let rating_u32 = (rating * 10.0).round() as u32;
        users.insert(user);
        movies.insert(movie);
        ratings.insert(Rating {
            user,
            movie,
            rating_u32,
        });
    }

    Ok((users, movies, ratings))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Data CSV file from https://www.kaggle.com/c/eecs498/data
    #[arg(long)]
    ratings_csv: String,

    // Number of epochs to train
    #[arg(long, default_value = "250")]
    epochs: u32,

    // Learning rate
    #[arg(long, default_value = "0.01")]
    lr: f32,

    // Regularization factor
    #[arg(long, default_value = "0.01")]
    reg: f32,

    // Number of features
    #[arg(long, default_value = "100")]
    n_features: usize,
}

fn mean_normalization(ratings: &Tensor, R: &Tensor) -> Result<Tensor> {
    let sum = ratings.mul(&R)?.sum(1)?;
    let count = R.sum(1)?;
    let mean = sum.div(&count)?;
    let adjusted = ratings.broadcast_sub(&mean.unsqueeze(1)?)?;
    Ok(adjusted)
}

fn cost(X: &Tensor, W: &Tensor, Y: &Tensor, R: &Tensor) -> Result<f32> {
    let c = X
        .matmul(&W.t()?)?
        .mul(&R)?
        .sub(&Y.mul(&R)?)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok(c)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let reg = Tensor::new(args.reg, &Device::cuda_if_available(0)?)?;
    let lr = Tensor::new(args.lr, &Device::cuda_if_available(0)?)?;

    let device = Device::cuda_if_available(0)?;

    let (users, movies, ratings) = load_ratings(&args.ratings_csv).unwrap();
    let mut users: Vec<u32> = users.into_iter().collect();
    users.sort();

    let mut movies: Vec<u32> = movies.into_iter().collect();
    movies.sort();

    let mut ratings: Vec<Rating> = ratings.into_iter().collect();
    ratings.sort();

    let n_users = users.len();
    let n_movies = movies.len();

    println!("n_users: {}, n_movies: {}", n_users, n_movies);

    let mut Y = vec![vec![-1.0; n_users as usize]; n_movies as usize];
    let mut R = vec![vec![0.0; n_users as usize]; n_movies as usize];

    for rating in ratings.iter() {
        let i = movies.iter().position(|&x| x == rating.movie).unwrap();
        let j = users.iter().position(|&x| x == rating.user).unwrap();
        Y[i][j] = rating.rating();
        R[i][j] = 1.0;
    }
    let R = R.iter().flatten().copied().collect::<Vec<f32>>();
    let R = Tensor::from_slice(&R, (n_movies, n_users), &device)?;

    let Y = Y.iter().flatten().copied().collect::<Vec<f32>>();
    let Y = Tensor::from_slice(&Y, (n_movies, n_users), &device)?;
    let Y = mean_normalization(&Y, &R)?;

    let mut X = Tensor::randn(0f32, 0.1, (n_movies, args.n_features), &device)?;
    let mut W = Tensor::randn(0f32, 0.1, (n_users, args.n_features), &device)?;

    for i in 0..args.epochs {
        let diff = X.matmul(&W.t()?)?.mul(&R)?.sub(&Y.mul(&R)?)?;
        let grad_X = diff.matmul(&W)?.add(&X.broadcast_mul(&reg)?)?;
        let grad_W = diff.t()?.matmul(&X)?.add(&W.broadcast_mul(&reg)?)?;

        X = X.sub(&grad_X.broadcast_mul(&lr)?)?;
        W = W.sub(&grad_W.broadcast_mul(&lr)?)?;

        println!("Epoch: {}, Cost: {}", i, cost(&X, &W, &Y, &R)?);
    }

    Ok(())
}
