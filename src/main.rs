#![feature(slice_as_array)]
mod math;
mod layers;

use rand::Rng;
use math::tensor::Tensor;
use layers::{prec::*,relu::*};

struct TrainingData{
    i: Tensor<f32,1>,
    o: Tensor<f32,1>
}

//x = expectation, r = reality
fn reconstruction_loss(x: f32, r: f32) -> f32{
    let diff = r - x;
    diff * diff
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
            o: Tensor::<f32,1>::values([1], &vec![1.])?,
        },
    ];

    let mut rng = rand::rng();
    let mut perc: PercLayer<f32> = PercLayer {
        w: Tensor::values([1, 2], &vec![rng.random_range(0.01..1.0), rng.random_range(0.01..1.0)])?,
        b: Tensor::values([1],&vec![rng.random_range(0.01..1.0)])?,
        saved_x: Tensor::zeroes([2]),
    };

    let mut relu: ReluLayer<f32> = ReluLayer{
        saved_x: Tensor::zeroes([1]),
    };

    println!("Sample 0(pre_training): {0}",relu.forwards(&perc.forwards(&training_data[0].i)?)?);
    println!("Sample 1(pre_training): {0}",relu.forwards(&perc.forwards(&training_data[1].i)?)?);
    println!("Sample 2(pre_training): {0}",relu.forwards(&perc.forwards(&training_data[2].i)?)?);
    println!("Sample 3(pre_training): {0}",relu.forwards(&perc.forwards(&training_data[3].i)?)?);

    const EPOCHS: usize = 1000;
    const LEARNING_RATE: f32 = 0.01;

    let mut sum_loss = 0.;

    for i in 0..EPOCHS {
        println!("Epoch {0}, avg loss: {1}",i,sum_loss / training_data.len() as f32);
        sum_loss = 0.;
        for d in &training_data {
            let forwards = relu.forwards(&perc.forwards(&d.i)?)?;

            let loss = reconstruction_loss(forwards[[0]], d.o[[0]]);

            sum_loss += loss;

            let grad_output = Tensor::values([1],&vec![2.0 * (forwards[[0]] - d.o[[0]])])?;

            let mut backwards = perc.backwards(&relu.backwards(&grad_output)?)?;

            backwards.w_grad.mul_scalar(LEARNING_RATE);
            backwards.b_grad.mul_scalar(LEARNING_RATE);

            let new_weights = (&perc.w - &backwards.w_grad)?;
            let new_biases = (&perc.b - &backwards.b_grad)?;

            perc.w = new_weights;
            perc.b = new_biases;
        }
    }

    println!("Sample 0(post_training): {0}",relu.forwards(&perc.forwards(&training_data[0].i)?)?);
    println!("Sample 1(post_training): {0}",relu.forwards(&perc.forwards(&training_data[1].i)?)?);
    println!("Sample 2(post_training): {0}",relu.forwards(&perc.forwards(&training_data[2].i)?)?);
    println!("Sample 3(post_training): {0}",relu.forwards(&perc.forwards(&training_data[3].i)?)?);

    println!("weights: {0}",perc.w);
    println!("biases: {0}",perc.b);

    Ok(())
}
