def q31_narrow_even(x: tensor[q31,32]) -> tensor[q15,32]:
  return narrow(x, to=q15, rounding=nearest_even, overflow=wrap)
