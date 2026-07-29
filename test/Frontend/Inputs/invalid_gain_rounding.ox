def invalid_gain_rounding(input: tensor[q15,64]) -> tensor[q15,64]:
  return gain(input, gain=19661, rounding=toward_negative)
