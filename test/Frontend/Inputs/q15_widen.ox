def q15_widen(x: tensor[q15,32]) -> tensor[q31,32]:
  return widen(x, to=q31)
