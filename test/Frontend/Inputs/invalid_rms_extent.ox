def invalid_rms_extent(input: tensor[q15,48]) -> tensor[q15,1]:
  return rms(input)
