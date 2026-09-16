def q15_infix_calls(x: tensor[q15,32], y: tensor[q15,32], z: tensor[q15,32]) -> tensor[q15,32]:
  return sub(add(x, mult(y, z)), shift(x, amount=-1))
