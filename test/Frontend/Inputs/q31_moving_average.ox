def q31_moving_average(x: tensor[q31,16]) -> tensor[q31,13]:
  return moving_average(x, window=4)
