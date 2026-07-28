def invalid_gain_shape(input: tensor[q15,64]) -> tensor[q15,32]:
  return gain(input, gain=19661)
