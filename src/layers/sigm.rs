use num_traits::Float;

use crate::math::tensor::*;

fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + T::exp(-x))
}

fn sigmoid_der<T: Float>(x:T) -> T {
    sigmoid(x) * (T::one() - sigmoid(x))
}

pub struct SigmoidLayer<T: Float> {
    pub saved_x: Tensor<T,1>
}

impl<T: Float> SigmoidLayer<T> {
    pub fn forwards(&mut self,x: &Tensor<T,1>) -> Result<Tensor<T,1>,Box<dyn std::error::Error>> {
        self.saved_x = x.clone();

        let mut newtensor = Tensor::<T,1>::zeroes(match x.shape().as_array(){
            Some(v) => *v,
            None => return Err(Box::new(TensorError::OprationFailed)) 
        });

        for (s,d) in x.data.iter().zip(&mut newtensor.data) {
            *d = sigmoid::<T>(*s);
        }

        Ok(newtensor)
    }

    pub fn backwards(&mut self,grad: &Tensor<T,1>) -> Result<Tensor<T,1>,Box<dyn std::error::Error>>{
        let mut newtensor = Tensor::<T,1>::zeroes(match self.saved_x.shape().as_array(){
            Some(v) => *v,
            None => return Err(Box::new(TensorError::OprationFailed)) 
        });

        for ((g,d),s) in grad.data.iter().zip(&mut newtensor.data).zip(&self.saved_x.data) {
            *d = *g * sigmoid_der::<T>(*s);
        }

        Ok(newtensor)
    }
}