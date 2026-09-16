def invalid_widen_target(x: tensor[q31,32]) -> tensor[q15,32]:
  return widen(x, to=q15)
