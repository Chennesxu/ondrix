def f32_rms(x: tensor[f32,10]) -> tensor[f32,1]:
  return rms(x, contract=off)
