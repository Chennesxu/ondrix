def invalid_elementwise_mixed_width(x: tensor[q31,32], y: tensor[q15,32]) -> tensor[q31,32]:
  return add(x, y)
