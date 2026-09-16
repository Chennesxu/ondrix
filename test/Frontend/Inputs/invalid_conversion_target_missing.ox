def invalid_conversion_target_missing(x: tensor[q15,32]) -> tensor[q31,32]:
  return widen(x)
