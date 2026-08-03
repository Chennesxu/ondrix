def f32_fir_interpolate(
    input: tensor[f32,4], coefficients: tensor[f32,3]) -> tensor[f32,9]:
  return fir_interpolate(input, coefficients, factor=2, contract=off)
