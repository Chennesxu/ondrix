def q31_magnitude_floor(input: tensor[q31,16]) -> tensor[q31,9]:
  return magnitude(rfft(input), input_rounding = toward_negative, root_rounding = toward_negative)
