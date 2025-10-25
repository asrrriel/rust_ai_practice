use std::usize;
use std::ops::Add;
use std::ops::Mul;
use strum_macros::Display;
use num_traits::Float;

#[derive(Display,Debug)]
pub enum TensorError {
    WrongDataSize,
    ShapeMisMatch
}

impl std::error::Error for TensorError {}

pub struct Tensor<T: Float,const R: usize> {
    rank: usize,
    dimensionality: [usize; R],
    pub data: Vec<T>
}

impl<T: Float + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign,const R: usize> Tensor<T,R>{

    pub fn zeroes(dim: [usize; R]) -> Tensor<T, R> {
        let total_size = dim.iter().product();
        let vec: Vec<T> = vec![T::zero(); total_size];

        Tensor {
            rank: R,
            dimensionality: dim,
            data: vec
        }
    }

    pub fn values(dim: [usize; R], val: &Vec<T>) -> Result<Tensor<T,R>,Box<dyn std::error::Error>> {
        if val.len() != dim.iter().product() {
            return Err(Box::new(TensorError::WrongDataSize))
        }

        Ok(Tensor {
            rank: R,
            dimensionality: dim,
            data: val.clone()
        })
    }

    pub fn add_scalar(&mut self,s: T) {
        for v in &mut self.data {
            *v += s;
        }
    }

    pub fn sub_scalar(&mut self,s: T) {
        for v in &mut self.data {
            *v -= s;
        }
    }

    pub fn mul_scalar(&mut self,s: T) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    pub fn div_scalar(&mut self,s: T) {
        for v in &mut self.data {
            *v /= s;
        }
    }
}

impl<T: Float + std::ops::AddAssign,const R: usize> Add for Tensor<T,R> {
    type Output = Result<Tensor<T,R>,Box<dyn std::error::Error>>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rank != rhs.rank || self.dimensionality != rhs.dimensionality{
            return Err(Box::new(TensorError::ShapeMisMatch))
        }

        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor{
            rank: self.rank,
            dimensionality: self.dimensionality,
            data: data
        })
    }
}

impl<T: Float + std::ops::AddAssign,const R: usize> Mul for Tensor<T,R> {
    type Output = Result<Tensor<T,R>,Box<dyn std::error::Error>>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.rank != rhs.rank || self.dimensionality != rhs.dimensionality{
            return Err(Box::new(TensorError::ShapeMisMatch))
        }

        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(Tensor{
            rank: self.rank,
            dimensionality: self.dimensionality,
            data: data
        })
    }
}