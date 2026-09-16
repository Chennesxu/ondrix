def invalid_ratio_policy(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return ratio(x, y, nonpositive=zero)
