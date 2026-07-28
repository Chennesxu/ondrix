def invalid_gain_constant(input: tensor[q15,64]) -> tensor[q15,64]:
  return gain(input, gain=40000)
