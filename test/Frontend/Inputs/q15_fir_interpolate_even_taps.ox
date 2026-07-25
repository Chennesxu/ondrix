def q15_fir_interpolate_even_taps(
    input: tensor[q15,4], coefficients: tensor[q15,4]) -> tensor[q15,10]:
  return fir_interpolate(input, coefficients, factor=2)
