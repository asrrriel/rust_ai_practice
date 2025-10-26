use super::{Tensor,TrainingData};

use std::{fs, io, io::Read};

pub fn load_idx_data<const R: usize>(path: &str) -> Result<Tensor<f32,R>,Box<dyn std::error::Error>> {

    let mut f = fs::File::open(path)?;

    let mut buf = Vec::<u8>::new();
    f.read_to_end(&mut buf)?;

    if buf.len() < 4+(R*4) || buf[0..2] != [0; 2] {
        return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, "No header")));
    }

    if buf[2] != 0x8 {
        return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, "Not unsigned 8-bit integers")));
    }

    if usize::from(buf[3]) != R {
        return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, "Bad rank")));
    }

    let mut shape = [0; R];

    for i in 0..R {
        shape[i] = u32::from_be_bytes(buf[4+(i*4)..4+((i+1)*4)].try_into()?) as usize;
    }

    let mut newtensor = Tensor::<f32,R>::zeroes(shape);

    if buf.len() < 4+(R*4)+newtensor.data.len() {
        return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, "File too short")));
    }

    let mut byte_num = 4+(R*4);

    for i in 0..newtensor.data.len() {
        newtensor.data[i] = buf[byte_num] as f32;
        byte_num += 1;
    }

    Ok(newtensor)
}



pub fn load_mnist_train<const N: usize>(image_path: &str,label_path: &str) ->  Result<Vec<TrainingData>,Box<dyn std::error::Error>>{
    let images: Tensor<f32, 3> = load_idx_data(image_path)?;
    let labels: Tensor<f32, 1> = load_idx_data(label_path)?;

    if images.data.len() != 784*N || labels.data.len() != N {
        return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, "Wrong data size")));
    }

    let mut out = vec![
        TrainingData { 
            i: Tensor { rank: 0, shape: [0], data: vec![0.0], transposed: false },
            o: Tensor { rank: 0, shape: [0], data: vec![0.0], transposed: false } 
        }; N];

    for num in 0..N {
        let i: [f32; 784] = *images.data[num*784..(num+1)*784].as_array().unwrap();

        let mut o: [f32; 10] = [0.;10];
        o[labels.data[num] as usize] = 1.;

        out[num] = TrainingData {
             i: Tensor::values([784], &Vec::from(i)).expect("Creating input tensor failed"),
             o: Tensor::values([10], &Vec::from(o)).expect("Creating output tensor failed")
        };
    };

    Ok(out)
}