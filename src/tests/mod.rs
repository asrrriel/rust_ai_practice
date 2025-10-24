use super::math::tensor::*;

#[test]
fn test_zeroes() {
    let t: Tensor<f32,1> =  Tensor::zeroes([3]);

    assert_eq!(t.data,[0.,0.,0.]);

    let _2d: Tensor<f32,2> =  Tensor::zeroes([3,2]);

    assert_eq!(_2d.data,[0.,0.,0.,0.,0.,0.]);
}

#[test]
fn test_values() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.];

    let t: Tensor<f32,1> =  Tensor::values([3],&val)?;

    assert_eq!(t.data,val);

    let val_2d = vec![0.,1.,2.,3.,4.,5.];

    let _2d: Tensor<f32,2> =  Tensor::values([3,2],&val_2d)?;

    assert_eq!(_2d.data,val_2d);

    Ok(())
}

#[test]
fn test_add_scalar() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.];
    let val2 = vec![2.,1.,0.];

    let t: Tensor<f32,1> =  Tensor::values([3],&val)?;

    let t2: Tensor<f32,1> =  Tensor::values([3],&val2)?;

    let t3 = (t + t2)?;

    assert_eq!(t3.data,[2.,2.,2.]);

    Ok(())
}