def q31_rms_floor(input: tensor[q31,64]) -> tensor[q31,1]:
  return rms(input, input_rounding=toward_negative, root_rounding=toward_negative)
