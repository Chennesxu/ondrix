def invalid_unused_parameter(signal: tensor[q15,16], extra: tensor[q15,4]) -> tensor[q15,9]:
  return magnitude(rfft(signal))
