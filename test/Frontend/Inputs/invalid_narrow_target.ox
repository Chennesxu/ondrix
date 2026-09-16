def invalid_narrow_target(x: tensor[q15,32]) -> tensor[q15,32]:
  return narrow(x, to=q15)
