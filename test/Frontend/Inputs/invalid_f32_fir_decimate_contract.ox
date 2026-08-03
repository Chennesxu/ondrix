def invalid_fir_decimate_f32(input: tensor[f32,48], coefficients: tensor[f32,8]) -> tensor[f32,21]:
  return fir_decimate(input, coefficients, factor=2)
