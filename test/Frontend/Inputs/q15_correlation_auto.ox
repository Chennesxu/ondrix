def q15_correlation_auto(
    input: tensor[q15,6], kernel: tensor[q15,3]) -> tensor[q15,4]:
  return correlation(input, kernel)
