def invalid_fir_filter_dynamic_default(
    input: tensor[q15], coefficients: tensor[q15]) -> tensor[q15]:
  return fir_filter(input, coefficients, boundary=valid)
