use super::tensor::*;

#[test]
fn tensor_zeroes() -> Result<(),Box<dyn std::error::Error>> {
    let t =  Tensor::zeroes(1, vec![3])?;
    assert_eq!(t.data,[0.,0.,0.]);

    let _2d =  Tensor::zeroes(2, vec![3,2])?;
    assert_eq!(_2d.data,[0.,0.,0.,0.,0.,0.]);

    Ok(())
}

#[test]
fn tensor_values() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.];
    let t =  Tensor::values(1, vec![3],&val)?;
    assert_eq!(t.data,val);

    let val_2d = vec![0.,1.,2.,3.,4.,5.];
    let _2d =  Tensor::values(2, vec![3,2],&val_2d)?;
    assert_eq!(_2d.data,val_2d);

    Ok(())
}

#[test]
fn tensor_add_scalar() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.];
    let mut t =  Tensor::values(1, vec![3],&val)?;

    t.add_scalar(2.);
    assert_eq!(t.data,[2.,3.,4.]);

    let val_2d = vec![0.,1.,2.,3.,4.,5.];
    let mut _2d =  Tensor::values(2, vec![3,2],&val_2d)?;

    _2d.add_scalar(2.);
    assert_eq!(_2d.data,[2.,3.,4.,5.,6.,7.]);

    Ok(())
}

#[test]
fn tensor_sub_scalar() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.];
    let mut t =  Tensor::values(1, vec![3],&val)?;

    t.sub_scalar(2.);
    assert_eq!(t.data,[-2.,-1.,0.]);

    let val_2d = vec![0.,1.,2.,3.,4.,5.];
    let mut _2d=  Tensor::values(2, vec![3,2],&val_2d)?;

    _2d.sub_scalar(2.);
    assert_eq!(_2d.data,[-2.,-1.,0.,1.,2.,3.]);

    Ok(())
}

#[test]
fn tensor_mul_scalar() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.];
    let mut t =  Tensor::values(1, vec![3],&val)?;

    t.mul_scalar(2.);
    assert_eq!(t.data,[0.,2.,4.]);

    let val_2d = vec![0.,1.,2.,3.,4.,5.];
    let mut _2d =  Tensor::values(2,vec![3,2],&val_2d)?;

    _2d.mul_scalar(2.);
    assert_eq!(_2d.data,[0.,2.,4.,6.,8.,10.]);

    Ok(())
}

#[test]
fn tensor_div_scalar() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,2.,4.];
    let mut t=  Tensor::values(1,vec![3],&val)?;

    t.div_scalar(2.);
    assert_eq!(t.data,[0.,1.,2.]);

    let val_2d = vec![0.,2.,4.,6.,8.,10.];
    let mut _2d =  Tensor::values(2,vec![3,2],&val_2d)?;

    _2d.div_scalar(2.);
    assert_eq!(_2d.data,[0.,1.,2.,3.,4.,5.]);

    Ok(())
}

#[test]
fn tensor_add() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.];
    let val2 = vec![2.,1.,0.];
    let t =  Tensor::values(1,vec![3],&val)?;
    let t2 =  Tensor::values(1,vec![3],&val2)?;

    let t3 = (&t + &t2)?;
    assert_eq!(t3.data,[2.,2.,2.]);

    Ok(())
}

#[test]
fn tensor_shape_mismatch() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![5.,4.,2.];
    let val2 = vec![2.,1.,3.,4.];
    let t =  Tensor::values(1,vec![3],&val)?;
    let t2 =  Tensor::values(1,vec![4],&val2)?;

    let t3 = &t + &t2;
    assert!(t3.is_err());
    assert!(match t3 {
        Err(e) => e == TensorError::ShapeMismatch,
        _ => false
    });

    Ok(())
}

#[test]
fn tensor_index() -> Result<(),Box<dyn std::error::Error>> {
    let val_2d = vec![0.,1.,2.,3.,4.,5.];
    let mut _2d =  Tensor::values(2,vec![2,3],&val_2d)?;
    assert_eq!(_2d[&[0,0]],0.);
    assert_eq!(_2d[&[0,1]],1.);
    assert_eq!(_2d[&[0,2]],2.);
    assert_eq!(_2d[&[1,0]],3.);
    assert_eq!(_2d[&[1,1]],4.);
    assert_eq!(_2d[&[1,2]],5.);

    Ok(())
}

#[test]
fn tensor_index_oob() -> Result<(),Box<dyn std::error::Error>> {
    let val_2d = vec![0.,1.,2.,3.,4.,5.];
    let mut _2d =  Tensor::values(2, vec![2,3],&val_2d)?;
    assert!(_2d.get(vec![0,10]).is_err());

    Ok(())
}

#[test]
fn tensor_matmul() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.,3.,4.,5.];
    let val2 = vec![0.,1.,2.,3.,4.,5.];
    let t =  Tensor::values(2,vec![3,2],&val)?;
    let t2 =  Tensor::values(2,vec![2,3],&val2)?;

    let t3 = t.matmul(&t2)?;
    assert_eq!(t3.data,[3.,4.,5.,9.,14.,19.,15.,24.,33.]);

    Ok(())
}

#[test]
fn tensor_matmul_batched() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.,3.,4.,5.,0.,1.,2.,3.,4.,5.];
    let val2 = vec![0.,1.,2.,3.,4.,5.,0.,1.,2.,3.,4.,5.];
    let t =  Tensor::values(2,vec![2,3,2],&val)?;
    let t2 =  Tensor::values(2,vec![2,2,3],&val2)?;

    let t3 = t.matmul(&t2)?;
    assert_eq!(t3.data,[3.,4.,5.,9.,14.,19.,15.,24.,33.,3.,4.,5.,9.,14.,19.,15.,24.,33.,]);

    Ok(())
}
#[test]
fn tensor_transposed() -> Result<(),Box<dyn std::error::Error>> {
    let val_2d = vec![0.,1.,2.,3.,4.,5.];
    let mut _2d=  Tensor::values(2,vec![2,3],&val_2d)?;

    _2d.transpose();

    assert_eq!(_2d[&[0,0]],0.);
    assert_eq!(_2d[&[1,0]],1.);
    assert_eq!(_2d[&[2,0]],2.);
    assert_eq!(_2d[&[0,1]],3.);
    assert_eq!(_2d[&[1,1]],4.);
    assert_eq!(_2d[&[2,1]],5.);

    Ok(())
}

#[test]
fn tensor_transposed_matmul() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.,3.,4.,5.];
    let val2 = vec![0.,3.,1.,4.,2.,5.];
    let t =  Tensor::values(2,vec![3,2],&val)?;
    let mut t2 =  Tensor::values(2,vec![3,2],&val2)?;

    t2.transpose();

    let t3 = t.matmul(&t2)?;
    assert_eq!(t3.data,[3.,4.,5.,9.,14.,19.,15.,24.,33.]);

    Ok(())
}

#[test]
fn tensor_transposed_add() -> Result<(),Box<dyn std::error::Error>> {
    let val = vec![0.,1.,2.,3.,4.,5.];
    let val2 = vec![5.,3.,1.,4.,2.,0.];
    let t =  Tensor::values(2,vec![3,2],&val)?;
    let mut t2 =  Tensor::values(2,vec![2,3],&val2)?;

    t2.transpose();

    let t3 = (&t + &t2)?;

    let expected = vec![
        t.get(vec![0,0])? + t2.get(vec![0,0])?,
        t.get(vec![0,1])? + t2.get(vec![0,1])?,
        t.get(vec![1,0])? + t2.get(vec![1,0])?,
        t.get(vec![1,1])? + t2.get(vec![1,1])?,
        t.get(vec![2,0])? + t2.get(vec![2,0])?,
        t.get(vec![2,1])? + t2.get(vec![2,1])?,
    ];
    assert_eq!(t3.data, expected);

    Ok(())
}