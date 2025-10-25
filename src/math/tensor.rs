use std::ops::AddAssign;
use std::usize;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Index;
use strum_macros::Display;
use num_traits::Float;

#[derive(Display,Debug,PartialEq,Eq)]
pub enum TensorError {
    WrongDataSize,
    ShapeMismatch,
    OutOfBounds,
    OprationFailed
}

impl std::error::Error for TensorError {}

pub struct Tensor<T: Float,const R: usize> {
    rank: usize,
    shape: [usize; R],
    pub data: Vec<T>
}

//constructors
impl<T: Float,const R: usize> Tensor<T,R>{
    pub fn zeroes(dim: [usize; R]) -> Tensor<T, R> {
        let total_size = dim.iter().product();
        let vec: Vec<T> = vec![T::zero(); total_size];

        Tensor {
            rank: R,
            shape: dim,
            data: vec
        }
    }

    pub fn values(dim: [usize; R], val: &Vec<T>) -> Result<Tensor<T,R>,TensorError> {
        if val.len() != dim.iter().product() {
            return Err(TensorError::WrongDataSize)
        }

        Ok(Tensor {
            rank: R,
            shape: dim,
            data: val.clone()
        })
    }
}

// scalar operations
impl<T: Float + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign,const R: usize> Tensor<T,R>{
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

//safe getter and setter
impl<T: Float,const R: usize> Tensor<T,R>{
    pub fn get(&self, index: [usize; R]) -> Result<&T,TensorError> {
        println!("{index:?}");
        let mut idx = 0;
        let mut stride = 1;

        for (a,b) in index.into_iter().zip(self.shape).rev(){
            if a >= b {
                return Err(TensorError::OutOfBounds)
            }
            idx += a * stride;
            stride *= b;
        };

        Ok(&self.data[idx])
    }

    pub fn set(&mut self, index: [usize; R], v: T) -> Result<(),TensorError> {
        println!("{index:?}");
        let mut idx = 0;
        let mut stride = 1;

        for (a,b) in index.into_iter().zip(self.shape).rev(){
            if a >= b {
                return Err(TensorError::OutOfBounds)
            }
            idx += a * stride;
            stride *= b;
        };

        (&mut self.data)[idx] = v;

        Ok(())
    }

}

//piecewise addition opeartor
impl<T: Float,const R: usize> Add for Tensor<T,R> {
    type Output = Result<Tensor<T,R>,TensorError>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rank != rhs.rank || self.shape != rhs.shape{
            return Err(TensorError::ShapeMismatch)
        }

        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor{
            rank: self.rank,
            shape: self.shape,
            data: data
        })
    }
}

//piecewise multiplication opeartor
impl<T: Float,const R: usize> Mul for Tensor<T,R> {
    type Output = Result<Tensor<T,R>,TensorError>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.rank != rhs.rank || self.shape != rhs.shape{
            return Err(TensorError::ShapeMismatch)
        }

        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(Tensor{
            rank: self.rank,
            shape: self.shape,
            data: data
        })
    }
}

//unsafe index opeartor
impl<T: Float,const R: usize> Index<[usize; R]> for Tensor<T,R> {
    type Output = T;

    fn index(&self, index: [usize; R]) -> &Self::Output {
        return match self.get(index){
            Ok(v) => v,
            Err(e) => panic!("Failed to index: {e}")
        }
    }
}

fn idx_to_coords(idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = idx;
    let mut result = vec![0; shape.len()];  
    for i in (0..shape.len()).rev(){
        result[i] = index % shape[i];
        index /= shape[i];
    }
    result
}

//matrix multiplication
impl<T: Float + AddAssign,const R: usize> Tensor<T,R>{
    pub fn matmul(&self, other: Tensor<T,R>) -> Result<Tensor<T,R>,TensorError> {
        //Rank has to be at or above 2 as matrix multiplication is undefined for vectors
        assert!(R >= 2, "Vectors(R=1) cannot be multiplied via matmul");

        //batch dimensions must be the exact same shape and M and N has to be 
        if self.shape[..R-2] != other.shape[..R-2] || self.shape[R - 1] != other.shape[R - 2] {
            return Err(TensorError::ShapeMismatch)
        }

        let mut newshape: [usize; R] = [0; R];

        newshape[..R-2].copy_from_slice(&self.shape[..R-2]);

        newshape[R-2] = self.shape[R-2];
        newshape[R-1] = other.shape[R-1];

        let batch_size = self.shape[..R-2].iter().product::<usize>();
        let mut newtensor = Tensor::zeroes(newshape);

        for b in 0..batch_size{
            let batch_coords = idx_to_coords(b, &self.shape[..R-2]);

            let mut c_coords: [usize; R] = {
                let mut tmp = [0; R];
                tmp[..R-2].copy_from_slice(&batch_coords);
                tmp[R-2] = 0;
                tmp[R-1] = 0;
                tmp
            };

            for i in 0..newshape[R-2] {
                for j in 0..newshape[R-1] {
                    let mut sum: T = T::zero();
                    for k in 0..self.shape[R-1] {

                        let a_coords: [usize; R] = {
                            let mut tmp = [0; R];
                            tmp[..R-2].copy_from_slice(&batch_coords);
                            tmp[R-2] = i;
                            tmp[R-1] = k;
                            tmp
                        };

                        let b_coords: [usize; R] = {
                            let mut tmp = [0; R];
                            tmp[..R-2].copy_from_slice(&batch_coords);
                            tmp[R-2] = k;
                            tmp[R-1] = j;
                            tmp
                        };

                        sum += *self.get(a_coords)? * *other.get(b_coords)?
                    } 
                    c_coords[R-2] = i;
                    c_coords[R-1] = j;
                    newtensor.set(c_coords,sum)?;
                } 
            } 
        }
        Ok(newtensor)
    }
}