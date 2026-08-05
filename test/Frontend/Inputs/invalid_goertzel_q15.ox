def invalid_goertzel_q15(x: tensor[q15,16]) -> tensor[q15,1]:
  return goertzel(x, bin=3, contract=off)
