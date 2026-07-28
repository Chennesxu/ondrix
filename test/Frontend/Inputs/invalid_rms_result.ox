def invalid_rms_result(input: tensor[q15,64]) -> tensor[q15,2]:
  return rms(input)
