#![feature(slice_as_array)]
mod math;
mod layers;

use core::panic;

use rand::Rng;
use math::tensor::Tensor;
use layers::{prec::*,sigm::*};

struct TrainingData{
    i: Tensor<f32,1>,
    o: Tensor<f32,1>
}

//x = expectation, r = reality
fn reconstruction_loss(x: f32, r: f32) -> f32{
    let diff = r - x;
    diff * diff * 0.5
}

//x = expectation, r = reality
fn reconstruction_loss_der(x: f32, r: f32) -> f32{
    x - r
}

struct SimplePreceptron {
    perc: PercLayer<f32>,
    sigm: SigmoidLayer<f32>,
    perc2: PercLayer<f32>,
    sigm2: SigmoidLayer<f32>,
    perc_grad: Option<PercGradient<f32>>,
    perc2_grad: Option<PercGradient<f32>>
}

impl SimplePreceptron {
    fn init() -> Result<SimplePreceptron, Box<dyn std::error::Error>> {
        let mut rng = rand::rng();
        let perc: PercLayer<f32> = PercLayer {
            w: Tensor::values([2, 2], &vec![rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0),rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])?,
            b: Tensor::values([2],&vec![rng.random_range(-1.0..1.0),rng.random_range(-1.0..1.0)])?,
            saved_x: Tensor::zeroes([2]),
        };

        let sigm: SigmoidLayer<f32> = SigmoidLayer{
            saved_x: Tensor::zeroes([2]),
        };

        let perc2: PercLayer<f32> = PercLayer {
            w: Tensor::values([1, 2], &vec![rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])?,
            b: Tensor::values([1],&vec![rng.random_range(-1.0..1.0)])?,
            saved_x: Tensor::zeroes([2]),
        };

        let sigm2: SigmoidLayer<f32> = SigmoidLayer{
            saved_x: Tensor::zeroes([1]),
        };

        Ok(SimplePreceptron { perc: perc, sigm: sigm, perc2: perc2, sigm2: sigm2, perc_grad: None, perc2_grad: None })
    }

    fn forwards(&mut self,x: &Tensor<f32,1>) -> Result<Tensor<f32, 1>, Box<dyn std::error::Error>> {
        Ok(self.sigm2.forwards(&self.perc2.forwards(&self.sigm.forwards(&self.perc.forwards(x)?)?)?)?)
    }

    fn backwards(&mut self,grad: &Tensor<f32,1>) -> Result<(), Box<dyn std::error::Error>> {
        self.perc2_grad = Some(self.perc2.backwards(&self.sigm2.backwards(grad)?)?);
        self.perc_grad = Some(self.perc.backwards(&self.sigm.backwards(&self.perc2_grad.as_ref().unwrap().o_grad)?)?);

        Ok(())
    }

    fn apply_grad(&mut self, learning_rate: f32) -> Result<(), Box<dyn std::error::Error>> {
        let perc_grad = match &mut self.perc_grad {
            Some(v) => v,
            None => {
                panic!("TODO: make this recoverable")
            }
        };

        let perc2_grad = match &mut self.perc2_grad {
            Some(v) => v,
            None => {
                panic!("TODO: make this recoverable")
            }
        };

        perc_grad.w_grad.mul_scalar(learning_rate);
        perc_grad.b_grad.mul_scalar(learning_rate);
        perc2_grad.w_grad.mul_scalar(learning_rate);
        perc2_grad.b_grad.mul_scalar(learning_rate);

        let new_weights = (&self.perc.w - &perc_grad.w_grad)?;
        let new_biases = (&self.perc.b - &perc_grad.b_grad)?;
        let new_weights2 = (&self.perc2.w - &perc2_grad.w_grad)?;
        let new_biases2 = (&self.perc2.b - &perc2_grad.b_grad)?;

        self.perc.w = new_weights;
        self.perc.b = new_biases;
        self.perc2.w = new_weights2;
        self.perc2.b = new_biases2;

        Ok(())
    }

}


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let training_data: [TrainingData; _] = [
        TrainingData{
            i: Tensor::<f32,1>::values([2], &vec![0.,0.])?,
            o: Tensor::<f32,1>::values([1], &vec![0.])?,
        },
        TrainingData{
            i: Tensor::<f32,1>::values([2], &vec![1.,0.])?,
            o: Tensor::<f32,1>::values([1], &vec![1.])?,
        },
        TrainingData{
            i: Tensor::<f32,1>::values([2], &vec![0.,1.])?,
            o: Tensor::<f32,1>::values([1], &vec![1.])?,
        },
        TrainingData{
            i: Tensor::<f32,1>::values([2], &vec![1.,1.])?,
            o: Tensor::<f32,1>::values([1], &vec![0.])?,
        },
    ];

    let mut model = SimplePreceptron::init()?;



    const EPOCHS: usize = 100000;
    const LEARNING_RATE: f32 = 0.01;

    let mut sum_loss = 0.;

    for i in 0..EPOCHS {
        println!("Epoch {0}, avg loss: {1}",i,sum_loss / training_data.len() as f32);
        println!("Sample 0: {0}",model.forwards(&training_data[0].i)?);
        println!("Sample 1: {0}",model.forwards(&training_data[1].i)?);
        println!("Sample 2: {0}",model.forwards(&training_data[2].i)?);
        println!("Sample 3: {0}\n",model.forwards(&training_data[3].i)?);

        sum_loss = 0.;
        for d in &training_data {
            let forwards = model.forwards(&d.i)?;

            let loss = reconstruction_loss(forwards[[0]], d.o[[0]]);

            sum_loss += loss;

            let grad_output = Tensor::values([1],&vec![reconstruction_loss_der(forwards[[0]], d.o[[0]])])?;

            model.backwards(&grad_output)?;

            model.apply_grad(LEARNING_RATE)?;
        }
    }

    println!("weights: {0}",model.perc.w);
    println!("biases: {0}",model.perc.b);

    println!("weights2: {0}",model.perc2.w);
    println!("biases2: {0}",model.perc2.b);

    Ok(())
}
