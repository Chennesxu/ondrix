def invalid_literal_gain_range(x: tensor[q15,16]) -> tensor[q15,16]:
  return x * 40000
