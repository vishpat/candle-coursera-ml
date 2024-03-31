extern crate csv;
use std::collections::HashSet;
use std::vec;

use anyhow::Result;
use candle_core::{Device, Tensor};
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
    #[arg(long, default_value = "10")]
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

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;

    let (users, movies, ratings) = load_ratings(&args.ratings_csv).unwrap();
    let users: Vec<u32> = users.into_iter().collect();
    let movies: Vec<u32> = movies.into_iter().collect();
    let ratings: Vec<Rating> = ratings.into_iter().collect();

    let n_users = users.len();
    let n_movies = movies.len();

    let mut Y = vec![vec![0.0; n_movies as usize]; n_users as usize];
    let mut R = vec![vec![0.0; n_movies as usize]; n_users as usize];

    for rating in ratings.iter() {
        let i = users.iter().position(|&x| x == rating.movie).unwrap();
        let j = movies.iter().position(|&x| x == rating.user).unwrap();
        Y[i][j] = rating.rating();
        R[i][j] = 1.0;
    }

    let Y = Y.iter().flatten().copied().collect::<Vec<f32>>();
    let Y = Tensor::from_slice(&Y, (n_movies, n_users), &device)?;

    let R = R.iter().flatten().copied().collect::<Vec<f32>>();
    let R = Tensor::from_slice(&R, (n_movies, n_users), &device)?;

    let X = Tensor::randn(0f32, 1., (n_movies, args.n_features), &device)?;
    let Theta = Tensor::randn(0f32, 1., (n_users, args.n_features), &device)?;

    Ok(())
}
