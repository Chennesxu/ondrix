def q15_literals(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  triple = 3 * y
  return triple + x * 16384 - 1024 - gain(x, gain=16384, rounding=nearest_even)
