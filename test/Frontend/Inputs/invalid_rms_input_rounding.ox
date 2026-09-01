def q15_rms_preshift(input: tensor[q15,64]) -> tensor[q15,1]:
  return rms(input, input_rounding=nearest_even)
