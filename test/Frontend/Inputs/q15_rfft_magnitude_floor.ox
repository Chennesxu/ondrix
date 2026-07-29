def q15_rfft_magnitude_floor(input: tensor[q15,16]) -> tensor[q15,9]:
  return magnitude(rfft(input), root_rounding=toward_negative)
