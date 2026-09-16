def q15_ratio_saturate(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return ratio(x, y, rounding=toward_zero, nonpositive=saturate)
