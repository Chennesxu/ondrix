def q15_gain(input: tensor[q15,64]) -> tensor[q15,64]:
  return gain(input, gain=19661)
