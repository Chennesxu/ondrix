def f32_fir_decimate(
    input: tensor[f32,12], coefficients: tensor[f32,5]) -> tensor[f32,4]:
  return fir_decimate(input, coefficients, factor=2, contract=fma)
