def q15_fir_auto(
    window: buffer[q15,4], coefficients: buffer[q15,4]) -> q15:
  return fir(window, coefficients)
