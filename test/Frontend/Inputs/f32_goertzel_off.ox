def f32_goertzel_off(x: tensor[f32,16]) -> tensor[f32,1]:
  return goertzel(x, bin=3, contract=off)
