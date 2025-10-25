use std::ops::AddAssign;
use num_traits::Float;

use crate::math::tensor::*;

pub struct PercLayer<T: Float> {
    pub w: Tensor<T,2>,
    pub b: Tensor<T,1>,
    pub saved_x: Tensor<T,1>
}

pub struct PercGradient<T: Float> {
    pub w_grad: Tensor<T,2>,
    pub b_grad: Tensor<T,1>
}

impl<T: Float + AddAssign> PercLayer<T> {
    pub fn forwards(&mut self,x: &Tensor<T,1>) -> Result<Tensor<T,1>,Box<dyn std::error::Error>> {
        self.saved_x = x.clone(); 
        Ok((&self.w.matmul_matvec(x)? + &self.b)?)
    }

    pub fn backwards(&mut self,grad: &Tensor<T,1>) -> Result<PercGradient<T>,Box<dyn std::error::Error>>{
        Ok(
            PercGradient{ 
                w_grad: grad.outer(&self.saved_x)?,
                b_grad: grad.clone()
            }
        )
    }
}