def invalid_rms_ties_positive(input: tensor[q15,64]) -> tensor[q15,1]:
  return rms(input, root_rounding=nearest_ties_positive)
