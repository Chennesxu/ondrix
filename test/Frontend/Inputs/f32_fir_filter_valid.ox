kernel f32_fir_filter_valid(input: tensor[f32], coefficients: tensor[f32]) -> tensor[f32]:
  return fir_filter(input, coefficients, boundary=valid, contract=fma)
