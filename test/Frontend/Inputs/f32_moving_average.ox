def f32_moving_average(x: tensor[f32,8]) -> tensor[f32,6]:
  return moving_average(x, window=3, contract=off)
