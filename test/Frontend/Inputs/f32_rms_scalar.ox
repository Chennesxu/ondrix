def f32_rms_scalar(x: buffer[f32,10]) -> f32:
  return rms(x, contract=fast)
