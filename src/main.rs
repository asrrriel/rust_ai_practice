#![feature(slice_as_array)]
mod math;
//mod loader;
//mod layers;
//
//use core::panic;
//
//use rand::Rng;
//use math::tensor::Tensor;
//use layers::{prec::*,sigm::*};
//
//use crate::loader::load_mnist_train;
//
//#[derive(Clone)]
//struct TrainingData{
//    i: Tensor<f32,1>,
//    o: Tensor<f32,1>
//}
//
////x = expectation, r = reality
//fn reconstruction_loss(x: f32, r: f32) -> f32{
//    let diff = r - x;
//    diff * diff * 0.5
//}
//
////x = expectation, r = reality
//fn reconstruction_loss_der(x: f32, r: f32) -> f32{
//    x - r
//}
//
//struct MNISTSolver {
//    perc: PercLayer<f32>,
//    sigm: SigmoidLayer<f32>,
//    perc2: PercLayer<f32>,
//    sigm2: SigmoidLayer<f32>,
//    perc_grad: Option<PercGradient<f32>>,
//    perc2_grad: Option<PercGradient<f32>>
//}
//
//impl MNISTSolver {
//    fn init() -> Result<MNISTSolver, Box<dyn std::error::Error>> {
//        let mut rng = rand::rng();
//
//        let mut percw = vec![0.; 784*64];
//
//        percw.fill_with(|| {
//            rng.random_range(-1.0..1.0)
//        });
//
//        let mut percb = vec![0.; 64];
//
//        percb.fill_with(|| {
//            rng.random_range(-1.0..1.0)
//        });
//
//        let mut perc2w = vec![0.; 64*10];
//
//        perc2w.fill_with(|| {
//            rng.random_range(-1.0..1.0)
//        });
//
//        let mut perc2b = vec![0.; 10];
//
//        perc2b.fill_with(|| {
//            rng.random_range(-1.0..1.0)
//        });
//
//        let perc: PercLayer<f32> = PercLayer {
//            w: Tensor::values([64, 784], &percw)?,
//            b: Tensor::values([64],&percb)?,
//            saved_x: Tensor::zeroes([784]),
//        };
//
//        let sigm: SigmoidLayer<f32> = SigmoidLayer{
//            saved_x: Tensor::zeroes([64]),
//        };
//
//        let perc2: PercLayer<f32> = PercLayer {
//            w: Tensor::values([10, 64], &perc2w)?,
//            b: Tensor::values([10],&perc2b)?,
//            saved_x: Tensor::zeroes([64]),
//        };
//
//        let sigm2: SigmoidLayer<f32> = SigmoidLayer{
//            saved_x: Tensor::zeroes([10]),
//        };
//
//        Ok(MNISTSolver { perc: perc, sigm: sigm, perc2: perc2, sigm2: sigm2, perc_grad: None, perc2_grad: None })
//    }
//
//    fn forwards(&mut self,x: &Tensor<f32,1>) -> Result<Tensor<f32, 1>, Box<dyn std::error::Error>> {
//        Ok(self.sigm2.forwards(&self.perc2.forwards(&self.sigm.forwards(&self.perc.forwards(x)?)?)?)?)
//    }
//
//    fn backwards(&mut self,grad: &Tensor<f32,1>) -> Result<(), Box<dyn std::error::Error>> {
//        self.perc2_grad = Some(self.perc2.backwards(&self.sigm2.backwards(grad)?)?);
//        self.perc_grad = Some(self.perc.backwards(&self.sigm.backwards(&self.perc2_grad.as_ref().unwrap().o_grad)?)?);
//
//        Ok(())
//    }
//
//    fn apply_grad(&mut self, learning_rate: f32) -> Result<(), Box<dyn std::error::Error>> {
//        let perc_grad = match &mut self.perc_grad {
//            Some(v) => v,
//            None => {
//                panic!("TODO: make this recoverable")
//            }
//        };
//
//        let perc2_grad = match &mut self.perc2_grad {
//            Some(v) => v,
//            None => {
//                panic!("TODO: make this recoverable")
//            }
//        };
//
//        perc_grad.w_grad.mul_scalar(learning_rate);
//        perc_grad.b_grad.mul_scalar(learning_rate);
//        perc2_grad.w_grad.mul_scalar(learning_rate);
//        perc2_grad.b_grad.mul_scalar(learning_rate);
//
//        let new_weights = (&self.perc.w - &perc_grad.w_grad)?;
//        let new_biases = (&self.perc.b - &perc_grad.b_grad)?;
//        let new_weights2 = (&self.perc2.w - &perc2_grad.w_grad)?;
//        let new_biases2 = (&self.perc2.b - &perc2_grad.b_grad)?;
//
//        self.perc.w = new_weights;
//        self.perc.b = new_biases;
//        self.perc2.w = new_weights2;
//        self.perc2.b = new_biases2;
//
//        Ok(())
//    }
//
//}


fn main() -> Result<(), Box<dyn std::error::Error>>{
//    let training_data = load_mnist_train::<10000>("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")?;
//
//    let mut model = MNISTSolver::init()?;
//
//    const EPOCHS: usize = 10;
//    const LEARNING_RATE: f32 = 0.01;
//
//    let mut sum_loss: f32 = 0.;
//
//    for i in 0..EPOCHS {
//        println!("Epoch {0}, avg loss: {1}",i,sum_loss / training_data.len() as f32);
//        sum_loss = 0.;
//        for d in &training_data {
//            let forwards = model.forwards(&d.i)?;
//
//            let loss = reconstruction_loss(forwards[[0]], d.o[[0]]);
//
//            sum_loss += loss;
//
//            let grad_output = Tensor::values([1],&vec![reconstruction_loss_der(forwards[[0]], d.o[[0]])])?;
//
//            model.backwards(&grad_output)?;
//
//            model.apply_grad(LEARNING_RATE)?;
//        }
//    }
//
//    println!("layer 1 weights: {0}",model.perc.w);
//    println!("layer 1 biases: {0}",model.perc.b);
//
//    println!("layer 2 weights: {0}",model.perc2.w);
//    println!("layer 2 biases: {0}",model.perc2.b);
//
    Ok(())
}
