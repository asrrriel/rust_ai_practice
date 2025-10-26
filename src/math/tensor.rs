use std::ops::AddAssign;
use std::usize;
use std::ops::Add;
use std::ops::Sub;
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

#[derive(Clone)]
pub struct Tensor<T: Float,const R: usize> {
    pub rank: usize,
    pub shape: [usize; R],
    pub data: Vec<T>,
    pub transposed: bool // swaps last 2 dimensions on index if true
}

//constructors
impl<T: Float,const R: usize> Tensor<T,R>{
    pub fn zeroes(dim: [usize; R]) -> Tensor<T, R> {
        let total_size = dim.iter().product();
        let vec: Vec<T> = vec![T::zero(); total_size];

        Tensor {
            rank: R,
            shape: dim,
            data: vec,
            transposed: false
        }
    }

    pub fn values(dim: [usize; R], val: &Vec<T>) -> Result<Tensor<T,R>,TensorError> {
        if val.len() != dim.iter().product() {
            return Err(TensorError::WrongDataSize)
        }

        Ok(Tensor {
            rank: R,
            shape: dim,
            data: val.clone(),
            transposed: false
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

    fn offset(&self, _index: [usize; R]) -> Result<usize,TensorError> {
        let index = match self.transposed {
            false => _index,
            true => {
                let mut temp = [0; R];

                temp.copy_from_slice(&_index[0..R]);

                let t = temp[R-1];
                temp[R-1] = temp[R-2];
                temp[R-2] = t;

                temp
            }
        };

        let mut idx = 0;
        let mut stride = 1;

        for (a,b) in index.into_iter().zip(self.shape).rev(){
            if a >= b {
                return Err(TensorError::OutOfBounds)
            }
            idx += a * stride;
            stride *= b;
        };

        Ok(idx)
    }

    pub fn shape(&self) -> Vec<usize> {
        match self.transposed {
            false => Vec::from(self.shape),
            true => {
                let mut temp = [0; R];

                temp.copy_from_slice(&self.shape[0..R]);

                let t = temp[R-1];
                temp[R-1] = temp[R-2];
                temp[R-2] = t;

                Vec::from(temp)
            }
        }
    }

    pub fn get(&self, index: [usize; R]) -> Result<&T,TensorError> {
        let idx = self.offset(index)?;
        Ok(&self.data[idx])
    }

    pub fn set(&mut self, index: [usize; R], v: T) -> Result<(),TensorError> {
        let idx = self.offset(index)?;
        (&mut self.data)[idx] = v;

        Ok(())
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

//piecewise addition opeartor
impl<'a,T: Float,const R: usize> Add<&'a Tensor<T, R>> for &'a Tensor<T,R> {
    type Output = Result<Tensor<T,R>,TensorError>;

    fn add(self, rhs: Self) -> Self::Output {
        let myshape = self.shape();
        let othershape = rhs.shape();

        if self.rank != rhs.rank || myshape != othershape{
            return Err(TensorError::ShapeMismatch)
        }

        let mut newtensor = Tensor::zeroes(*match myshape.as_array(){
            Some(v) => v,
            None => return Err(TensorError::OprationFailed) 
        });

        for i in 0..newtensor.data.len() {
            let index = *match idx_to_coords(i, &myshape).as_array() {
                Some(v) => v,
                None => return Err(TensorError::OprationFailed) 
            }; // map linear i to multi-dimensional index
            let a_idx = self.offset(index)?;
            let b_idx = rhs.offset(index)?;
            newtensor.data[i] = self.data[a_idx] + rhs.data[b_idx];
        }

        Ok(newtensor)
    }
}

//piecewise subtraction opeartor
impl<'a,T: Float,const R: usize> Sub<&'a Tensor<T, R>> for &'a Tensor<T,R> {
    type Output = Result<Tensor<T,R>,TensorError>;

    fn sub(self, rhs: Self) -> Self::Output {
        let myshape = self.shape();
        let othershape = rhs.shape();

        if self.rank != rhs.rank || myshape != othershape{
            return Err(TensorError::ShapeMismatch)
        }

        let mut newtensor = Tensor::zeroes(*match myshape.as_array(){
            Some(v) => v,
            None => return Err(TensorError::OprationFailed) 
        });

        for i in 0..newtensor.data.len() {
            let index = *match idx_to_coords(i, &myshape).as_array() {
                Some(v) => v,
                None => return Err(TensorError::OprationFailed) 
            }; // map linear i to multi-dimensional index
            let a_idx = self.offset(index)?;
            let b_idx = rhs.offset(index)?;
            newtensor.data[i] = self.data[a_idx] - rhs.data[b_idx];
        }

        Ok(newtensor)
    }
}

//piecewise multiplication opeartor
impl<'a,T: Float,const R: usize> Mul<&'a Tensor<T, R>> for &'a Tensor<T,R> {
    type Output = Result<Tensor<T,R>,TensorError>;

    fn mul(self, rhs: Self) -> Self::Output {
        let myshape = self.shape();
        let othershape = rhs.shape();

        if self.rank != rhs.rank || myshape != othershape{
            return Err(TensorError::ShapeMismatch)
        }

        let mut newtensor = Tensor::zeroes(*match myshape.as_array(){
            Some(v) => v,
            None => return Err(TensorError::OprationFailed) 
        });

        for i in 0..newtensor.data.len() {
            let index = *match idx_to_coords(i, &myshape).as_array() {
                Some(v) => v,
                None => return Err(TensorError::OprationFailed) 
            }; // map linear i to multi-dimensional index
            let a_idx = self.offset(index)?;
            let b_idx = rhs.offset(index)?;
            newtensor.data[i] = self.data[a_idx] * rhs.data[b_idx];
        }

        Ok(newtensor)
    }
}

//matrix multiplication
impl<T: Float + AddAssign,const R: usize> Tensor<T,R>{
    pub fn matmul(&self, other: &Tensor<T,R>) -> Result<Tensor<T,R>,TensorError> {
        //Rank has to be at or above 2 as matrix multiplication is undefined for vectors
        assert!(R >= 2, "Vectors(R=1) cannot be multiplied via matmul");

        let myshape: [usize; R] = match self.shape().as_array(){
            Some(v) => *v,
            None => return Err(TensorError::OprationFailed) 
        };

        let othershape: [usize; R] = match other.shape().as_array(){
            Some(v) => *v,
            None => return Err(TensorError::OprationFailed) 
        };

        //batch dimensions must be the exact same shape and M and N has to be 
        if myshape[..R-2] != othershape[..R-2] || myshape[R - 1] != othershape[R - 2] {
            return Err(TensorError::ShapeMismatch)
        }

        let mut newshape: [usize; R] = [0; R];

        newshape[..R-2].copy_from_slice(&self.shape[..R-2]);

        newshape[R-2] = myshape[R-2];
        newshape[R-1] = othershape[R-1];

        let batch_size = myshape[..R-2].iter().product::<usize>();
        let mut newtensor = Tensor::zeroes(newshape);

        for b in 0..batch_size{
            let batch_coords = idx_to_coords(b, &myshape[..R-2]);

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
                    for k in 0..myshape[R-1] {

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

//matrix-vector multiplication
impl<T: Float + AddAssign> Tensor<T,2>{
    pub fn matmul_matvec(&self, other: &Tensor<T,1>) -> Result<Tensor<T,1>,TensorError> {
        let myshape: [usize; 2] = match self.shape().as_array(){
            Some(v) => *v,
            None => return Err(TensorError::OprationFailed) 
        };

        let othershape: [usize; 1] = match other.shape().as_array(){
            Some(v) => *v,
            None => return Err(TensorError::OprationFailed) 
        };

        //the shared axis has to be there
        if myshape[1] != othershape[0] {
            return Err(TensorError::ShapeMismatch)
        }

        let newshape: [usize; 1] = [myshape[0]];

        let mut newtensor = Tensor::<T,1>::zeroes(newshape);

        for i in 0..newshape[0] {
            let mut sum: T = T::zero();
            for k in 0..myshape[1] {
                sum += *self.get([i,k])? * *other.get([k])?
            } 
            newtensor.set([i],sum)?;
        } 

        Ok(newtensor)
    }
}

//outer product
impl<T: Float> Tensor<T,1>{
    pub fn outer(&self, other: &Tensor<T,1>) -> Result<Tensor<T,2>,TensorError> {
        let myshape: [usize; 1] = match self.shape().as_array(){
            Some(v) => *v,
            None => return Err(TensorError::OprationFailed) 
        };

        let othershape: [usize; 1] = match other.shape().as_array(){
            Some(v) => *v,
            None => return Err(TensorError::OprationFailed) 
        };

        let newshape: [usize; 2] = [myshape[0],othershape[0]];
        let mut newtensor = Tensor::zeroes(newshape);

        for i in 0..newshape[0] {
            for j in 0..newshape[1] {
                newtensor.set([i,j],*self.get([i])? * *other.get([j])?)?;
            }
        }
        Ok(newtensor)
    }
}


//matrix transposition
impl<T: Float,const R: usize> Tensor<T,R>{
    pub fn transpose(&mut self) {
        //Rank has to be at or above 2 as transposition does jack for vectors
        assert!(R >= 2, "Vectors(R=1) cannot be transposed");

        self.transposed = !self.transposed
    }
}

impl<T: Float + std::fmt::Debug,const R: usize> std::fmt::Display for Tensor<T,R>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"{0:?}",self.data)
    }
}