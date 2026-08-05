def q15_sos_ties_positive(
    input: tensor[q15],
    coefficients: tensor[q15,1,5],
    scales: tensor[q15,1],
    state: tensor[q15,1,2])
    -> (tensor[q15], tensor[q15,1,2]):
  return sos_df2_fixed(
      input, coefficients, scales, state,
      accumulator=exact[40,saturate],
      state_rounding=nearest_ties_positive,
      state_overflow=saturate,
      output_rounding=nearest_ties_positive,
      output_overflow=saturate)
