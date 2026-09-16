def q31_narrow(x: tensor[q31,32]) -> tensor[q15,32]:
  return narrow(x, to=q15)
