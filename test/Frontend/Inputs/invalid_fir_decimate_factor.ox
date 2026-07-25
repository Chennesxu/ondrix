def invalid_fir_decimate_factor(
    input: tensor[q15,12], coefficients: tensor[q15,5]) -> tensor[q15,3]:
  return fir_decimate(input, coefficients, factor=3)
