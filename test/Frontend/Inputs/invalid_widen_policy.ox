def invalid_widen_policy(x: tensor[q15,32]) -> tensor[q31,32]:
  return widen(x, to=q31, rounding=nearest_even)
