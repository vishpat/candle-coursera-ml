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

fn main() {
    println!("Hello, world!");
}
