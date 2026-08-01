def q15_moving_average_odd(input: tensor[q15,40]) -> tensor[q15,38]:
  return moving_average(input, window=3)
