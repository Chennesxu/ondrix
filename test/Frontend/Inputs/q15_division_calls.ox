def q15_division_calls(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return sub(add(div(x, divisor=3), div(mult(y, x), divisor=4)), div(x, divisor=6, rounding=nearest_even))
