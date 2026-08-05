def invalid_goertzel_bin(x: tensor[f32,16]) -> tensor[f32,1]:
  return goertzel(x, bin=9, contract=off)
