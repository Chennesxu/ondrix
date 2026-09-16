def q15_literals(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  half = x * 16384
  return 3 * y + half - 1024 - gain(x, gain=16384, rounding=nearest_even)
