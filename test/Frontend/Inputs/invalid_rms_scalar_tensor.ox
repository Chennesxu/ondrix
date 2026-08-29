def f32_rms_tensor_arg(x: tensor[f32,10]) -> f32:
  return rms(x, contract=fast)
