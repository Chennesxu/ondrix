def invalid_sos_df2_fixed_layout(
    input: tensor[q15],
    coefficients: tensor[q15,5],
    scales: tensor[q15,1],
    state: tensor[q15,1,2])
    -> (tensor[q15], tensor[q15,1,2]):
  return sos_df2_fixed(
      input, coefficients, scales, state,
      accumulator=exact[40,saturate],
      state_rounding=nearest_even,
      state_overflow=saturate,
      output_rounding=toward_zero,
      output_overflow=wrap)
