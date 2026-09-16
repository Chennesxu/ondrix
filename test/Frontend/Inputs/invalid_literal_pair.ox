def invalid_literal_pair(x: tensor[q15,16]) -> tensor[q15,16]:
  return x + 1024 + 3 * 2
