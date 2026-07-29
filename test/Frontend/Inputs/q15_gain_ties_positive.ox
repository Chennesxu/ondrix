def q15_gain_ties_positive(input: tensor[q15,64]) -> tensor[q15,64]:
  return gain(input, gain=19661, rounding=nearest_ties_positive)
