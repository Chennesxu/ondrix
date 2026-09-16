def invalid_ratio_square(x: tensor[q15,32], y: tensor[q15,32], z: tensor[q15,32]) -> tensor[q15,32]:
  return x / (y * z + 1)
