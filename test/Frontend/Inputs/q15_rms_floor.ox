def q15_rms_floor(input: tensor[q15,64]) -> tensor[q15,1]:
  return rms(input, rounding=toward_negative)
