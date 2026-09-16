def q15_ratio(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return x / (y * y + 1)
