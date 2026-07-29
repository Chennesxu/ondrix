def q15_rms_bad(input: tensor[q15,64]) -> tensor[q15,1]:
  return rms(input, root_rounding=toward_zero)
