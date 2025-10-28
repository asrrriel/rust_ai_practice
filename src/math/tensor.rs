use std::usize;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Index;
use strum_macros::Display;
use blas::*;

#[derive(Display,Debug,PartialEq,Eq)]
pub enum TensorError {
    WrongDimSize,
    WrongDataSize,
    ShapeMismatch,
    OutOfBounds,
    OprationFailed
}

impl std::error::Error for TensorError {}

#[derive(Clone)]
pub struct Tensor {
    pub rank: usize,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub column_major: bool // swaps last 2 dimensions on index if true
}

//constructors
impl Tensor{
    pub fn zeroes(rank: usize,dim: Vec<usize>) -> Result<Tensor,TensorError> {
        if dim.len() != rank {
            return Err(TensorError::WrongDimSize)
        }
        let total_size = dim.iter().product();
        let vec: Vec<f64> = vec![0.; total_size];

        Ok(Tensor {
            rank: rank,
            shape: dim,
            data: vec,
            column_major: false
        })
    }

    pub fn values(rank: usize,dim: Vec<usize>, val: &Vec<f64>) -> Result<Tensor,TensorError> {
        if val.len() != dim.iter().product() {
            return Err(TensorError::WrongDataSize)
        }

        Ok(Tensor {
            rank: rank,
            shape: dim,
            data: val.clone(),
            column_major: false
        })
    }
}

// scalar operations
impl Tensor{
    pub fn add_scalar(&mut self,s: f64) {
        for v in &mut self.data {
            *v += s;
        }
    }

    pub fn sub_scalar(&mut self,s: f64) {
        for v in &mut self.data {
            *v -= s;
        }
    }

    pub fn mul_scalar(&mut self,s: f64) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    pub fn div_scalar(&mut self,s: f64) {
        for v in &mut self.data {
            *v /= s;
        }
    }

}

//safe getter and setter
impl Tensor{
    fn offset(&self, _index: Vec<usize>) -> Result<usize,TensorError> {
        if _index.len() != self.rank{
            return Err(TensorError::WrongDimSize)
        }

        let mut idx = 0;
        let mut stride = 1;

        let mut index = _index.clone();
        let mut factual_shape = self.shape.clone();

        if self.column_major {
            let t = index[self.rank-1];
            index[self.rank-1] = index[self.rank-2];
            index[self.rank-2] = t;

            let t = factual_shape[self.rank-1];
            factual_shape[self.rank-1] = factual_shape[self.rank-2];
            factual_shape[self.rank-2] = t;
        }


        for (a,b) in index.into_iter().zip(&factual_shape).rev(){
            if a >= *b {
                return Err(TensorError::OutOfBounds)
            }
            idx += a * stride;
            stride *= b;
        };

        Ok(idx)
    }

    pub fn get(&self, index: Vec<usize>) -> Result<&f64,TensorError> {
        let idx = self.offset(index)?;
        Ok(&self.data[idx])
    }

    pub fn set(&mut self, index: Vec<usize>, v: f64) -> Result<(),TensorError> {
        let idx = self.offset(index)?;
        (&mut self.data)[idx] = v;

        Ok(())
    }

}

//unsafe index opeartor
impl Index<&[usize]> for Tensor {
    type Output = f64;

    fn index(&self, index: &[usize]) -> &Self::Output {
        match self.get(Vec::from(index)){
            Ok(v) => v,
            Err(e) => panic!("Failed to index: {e}")
        }
    }
}

//piecewise addition opeartor
impl<'a> Add<&'a Tensor> for &'a Tensor {
    type Output = Result<Tensor,TensorError>;

    fn add(self, rhs: Self) -> Self::Output {
        let myshape = &self.shape;
        let othershape = &rhs.shape;

        if self.rank != rhs.rank || myshape != othershape{
            return Err(TensorError::ShapeMismatch)
        }

        let mut newtensor = Tensor::values(self.rank, self.shape.clone(),&rhs.data)?;

        unsafe {
            daxpy(self.data.len() as i32, 1., self.data.as_slice(), 1, newtensor.data.as_mut_slice(), 1);
        };

        Ok(newtensor)
    }
}

//piecewise subtraction opeartor
impl<'a> Sub<&'a Tensor> for &'a Tensor {
    type Output = Result<Tensor,TensorError>;

    fn sub(self, rhs: Self) -> Self::Output {
        let myshape = &self.shape;
        let othershape = &rhs.shape;

        if self.rank != rhs.rank || myshape != othershape{
            return Err(TensorError::ShapeMismatch)
        }

        let mut newtensor = Tensor::values(self.rank,self.shape.clone(), &self.data)?;

        unsafe {
            daxpy(self.data.len() as i32, -1., rhs.data.as_slice(), 1, newtensor.data.as_mut_slice(), 1);
        };

        Ok(newtensor)
    }
}

//matrix multiplication
impl Tensor{
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor,TensorError> {
        //Rank has to be at or above 2 as matrix multiplication is undefined for vectors
        assert!(self.rank == other.rank && self.rank >= 2, "Vectors(R=1) cannot be multiplied via matmul");

        //batch dimensions must be the exact same shape and M and N has to be 
        if self.shape[..self.rank-2] != other.shape[..self.rank-2] || self.shape[self.rank - 1] != other.shape[self.rank - 2] {
            return Err(TensorError::ShapeMismatch)
        }

        let mut newshape = Vec::<usize>::new();

        newshape.copy_from_slice(&self.shape[..self.rank-2]);

        newshape.push(self.shape[self.rank-2]);
        newshape.push(other.shape[self.rank-1]);

        let n_batches = self.shape[..self.rank-2].iter().product::<usize>();
        let batch_size_a = self.shape[self.rank-2] * self.shape[self.rank-1];
        let batch_size_b = other.shape[self.rank-2] * other.shape[self.rank-1];
        let batch_size_c = self.shape[self.rank-2] * other.shape[self.rank-1];

        let mut newtensor = Tensor::zeroes(self.rank,newshape)?;
        newtensor.column_major = true; //BLAS outputs column major

        for i in 0..n_batches {
            let offset_a = i * batch_size_a;
            let offset_b = i * batch_size_b;
            let offset_c = i * batch_size_c; // compute from M*N

            unsafe {
                dgemm(if self.column_major { b'N'} else { b'T'}, // data layout is row major by default so....
                      if other.column_major { b'N'} else { b'T'},
                           self.shape[self.rank-2] as i32, 
                           other.shape[self.rank-1] as i32, 
                           self.shape[self.rank-1] as i32, 
                       1.,
                           &self.data[offset_a..], 
                         self.shape[self.rank-1] as i32, 
                           &other.data[offset_b..], 
                         other.shape[other.rank-1] as i32,
                        0.0, 
                           &mut newtensor.data[offset_c..],
                         newtensor.shape[newtensor.rank-1] as i32
                );
            }
        }

        Ok(newtensor)
    }
}

//matrix-vector multiplication
impl Tensor{
    pub fn matmul_matvec(&self, other: &Tensor) -> Result<Tensor,TensorError> {
        //Rank has to be at or above 2 and the second one has to be a vector
        assert!(self.rank >= 2 , "Vectors(R=1) cannot be multiplied via matmul_matvec");
        assert!(other.rank == self.rank - 1, "Second argument must be a vector or batched vector");
        assert!(self.shape[..self.rank-2] == other.shape[..other.rank-1], "Batch dimensions must match");
        assert!(self.shape[self.rank-1] == other.shape[other.rank-1], "Matrix columns must equal vector length");

        let mut newshape = Vec::<usize>::new();
        newshape.copy_from_slice(&self.shape[..self.rank-2]);
        newshape.push(self.shape[self.rank-2]);

        let n_batches = self.shape[..self.rank-2].iter().product::<usize>();
        let batch_size_a = self.shape[self.rank-2] * self.shape[self.rank-1];
        let batch_size_b = other.shape[self.rank-1];
        let batch_size_c = self.shape[self.rank-2];

        let mut newtensor = Tensor::zeroes(other.rank,newshape)?;

        for i in 0..n_batches {
            let offset_a = i * batch_size_a;
            let offset_b = i * batch_size_b;
            let offset_c = i * batch_size_c; // compute from M*N

            unsafe {
                dgemv(if self.column_major { b'N'} else { b'T'}, // data layout is row major by default so....
                           self.shape[self.rank-2] as i32,
                           other.shape[self.rank-1] as i32,
                       1.,
                           &self.data[offset_a..], 
                         self.shape[self.rank-1] as i32, 
                           &other.data[offset_b..],
                         1,
                        0.0,
                           &mut newtensor.data[offset_c..],
                        1
                );
            }
        }

        Ok(newtensor)
    }
}

//outer product
impl Tensor{
    pub fn outer(&self, other: &Tensor) -> Result<Tensor,TensorError> {
        assert!(self.rank == other.rank, "Only vectors can have an outer product");
        assert!(self.shape[..self.rank-1] == other.shape[..other.rank-1], "Batch dimensions must match");

        let mut newshape = Vec::<usize>::new();
        newshape.copy_from_slice(&self.shape[..self.rank-1]);
        newshape.push(self.shape[self.rank-1]);
        newshape.push(other.shape[other.rank-1]);

        let n_batches = self.shape[..self.rank-1].iter().product::<usize>();
        let batch_size_a = self.shape[self.rank-1];
        let batch_size_b = other.shape[self.rank-1];
        let batch_size_c = self.shape[self.rank-1] * other.shape[self.rank-1];

        let mut newtensor = Tensor::zeroes(self.rank + 1,newshape)?;
        newtensor.column_major = true; // BLAS outputs column-major

        for i in 0..n_batches {
            let offset_a = i * batch_size_a;
            let offset_b = i * batch_size_b;
            let offset_c = i * batch_size_c; // compute from M*N

            unsafe {
                dger(self.shape[self.rank-1] as i32,
                     other.shape[self.rank-1] as i32,
                 1.,
                     &self.data[offset_a..],
                  1,
                     &other.data[offset_b..],
                  1,
                     &mut newtensor.data[offset_c..],
                   self.shape[self.rank-1] as i32
                );
            }
        }

        Ok(newtensor)
    }
}


//matrix transposition
impl Tensor{
    pub fn transpose(&mut self) {
        //Rank has to be at or above 2 as transposition does jack for vectors
        assert!(self.rank >= 2, "Vectors(R=1) cannot be column_major");

        self.column_major = !self.column_major;
        let t = self.shape[self.rank-1];
        self.shape[self.rank-1] = self.shape[self.rank-2];
        self.shape[self.rank-2] = t;
    }
}

impl std::fmt::Display for Tensor{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"{0:?}",self.data)
    }
}