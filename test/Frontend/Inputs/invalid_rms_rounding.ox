def q15_rms_bad(input: tensor[q15,64]) -> tensor[q15,1]:
  return rms(input, rounding=toward_zero)
