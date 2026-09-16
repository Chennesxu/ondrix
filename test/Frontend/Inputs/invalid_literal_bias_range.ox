def invalid_literal_bias_range(x: tensor[q15,16]) -> tensor[q15,16]:
  return x + 40000
