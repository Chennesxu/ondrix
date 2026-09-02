def q31_sos_df2_fixed(
    input: tensor[q31],
    coefficients: tensor[q31,1,5],
    scales: tensor[q31,1],
    state: tensor[q31,1,2])
    -> (tensor[q31], tensor[q31,1,2]):
  return sos_df2_fixed(
      input, coefficients, scales, state,
      accumulator=exact[64,saturate],
      state_rounding=nearest_even,
      state_overflow=saturate,
      output_rounding=toward_zero,
      output_overflow=wrap)
