def q15_cic_decimate_saturate(input: tensor[q15,64]) -> tensor[q15,8]:
  return cic_decimate(input, stages=3, rate=8, delay=2, state_overflow=saturate,
                      rounding=nearest_ties_positive)
