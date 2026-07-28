def invalid_moving_average_shape(input: tensor[q15,40]) -> tensor[q15,32]:
  return moving_average(input, window=8)
