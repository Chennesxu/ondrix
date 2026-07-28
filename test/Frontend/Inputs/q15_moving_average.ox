def q15_moving_average(input: tensor[q15,40]) -> tensor[q15,33]:
  return moving_average(input, window=8)
