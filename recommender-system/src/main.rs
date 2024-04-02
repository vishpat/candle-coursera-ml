extern crate csv;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashSet;
use std::vec;
use std::{cmp::Ordering, collections::HashMap};

use anyhow::Result;
use candle_core::{Device, Tensor, D};
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

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
struct MovieDistance {
    id: u32,
    distance: u32,
}

// Step 2: Implement `PartialOrd` and `Ord` for the struct.
impl PartialOrd for MovieDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MovieDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl MovieDistance {
    fn new(id: u32, distance: u32) -> Self {
        Self { id, distance }
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

    #[arg(long)]
    movies_csv: String,

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

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let diff = x1.sub(&x2)?;
    let dist = diff.sqr()?.sum_all()?.sqrt()?;
    Ok(dist)
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
    }

    // Load movie titles
    let mut rdr = csv::Reader::from_path(&args.movies_csv)?;
    let mut movie_titles = HashMap::new();
    for result in rdr.records() {
        let record = result?;
        let movie_id: u32 = record[0].parse()?;
        let title = record[1].to_string();
        movie_titles.insert(movie_id, title);
    }

    // Choose a random movie and find similar movies
    let mut rng = thread_rng();
    let random_movie_id = movies.choose(&mut rng).unwrap();
    println!("Random movie: {}", movie_titles[random_movie_id]);

    let random_movie_idx = movies.iter().position(|&x| x == *random_movie_id).unwrap();
    let random_index_tensor = Tensor::from_slice(&[random_movie_idx as u32], &[1], &device)?;
    let random_movie_features = X.index_select(&random_index_tensor, 0)?;

    let mut movie_distances: Vec<MovieDistance> = Vec::new();
    for i in 0..n_movies {
        let movie_index_tensor = Tensor::from_slice(&[i as u32], &[1], &device)?;
        let movie_features = X.index_select(&movie_index_tensor, 0)?;
        let dist = cdist(&random_movie_features, &movie_features)?;
        let dist = dist.to_scalar::<f32>()?;
        let movie_distance = MovieDistance::new(movies[i], (dist * 1000.0) as u32);
        movie_distances.push(movie_distance);
    }

    movie_distances.sort();
    for i in 0..10 {
        let movie_id = movie_distances[i].id;
        let distance = movie_distances[i].distance;
        println!(
            "{}: {} (distance: {})",
            i + 1,
            movie_titles[&movie_id],
            distance
        );
    }

    Ok(())
}
