def q15_fir_decimate(
    input: tensor[q15,12], coefficients: tensor[q15,5]) -> tensor[q15,4]:
  return fir_decimate(input, coefficients, factor=2)
