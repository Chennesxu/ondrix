def invalid_elementwise_extent(x: tensor[q15,32], y: tensor[q15,16]) -> tensor[q15,32]:
  return add(x, y)
