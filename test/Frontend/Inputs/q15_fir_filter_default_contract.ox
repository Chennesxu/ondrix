def q15_fir_filter_default_contract(
    input: tensor[q15,64], coefficients: tensor[q15,8]) -> tensor[q15,57]:
  return fir_filter(input, coefficients, boundary=valid)
