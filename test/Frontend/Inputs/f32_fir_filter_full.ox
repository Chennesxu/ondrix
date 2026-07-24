def f32_fir_filter_full(input: tensor[f32,5], coefficients: tensor[f32,3]) -> tensor[f32,7]:
  return fir_filter(input, coefficients, boundary=full, contract=fma)
