def invalid_q15_magnitude_input_rounding(input: tensor[q15,16]) -> tensor[q15,9]:
  return magnitude(rfft(input), input_rounding = toward_negative)
