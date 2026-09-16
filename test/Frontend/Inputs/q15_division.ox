def q15_division(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return x / 3 + y * x / 4 - div(x, divisor=6, rounding=nearest_even)
