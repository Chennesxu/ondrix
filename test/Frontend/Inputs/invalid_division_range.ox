def invalid_division_range(x: tensor[q15,16]) -> tensor[q15,16]:
  return div(x, divisor=32768)
