def invalid_conversion_implicit(x: tensor[q15,32], y: tensor[q31,32]) -> tensor[q31,32]:
  return x + y
