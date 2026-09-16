def q15_ratio_calls(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return ratio(x, offset(mult(y, y), bias=1))
