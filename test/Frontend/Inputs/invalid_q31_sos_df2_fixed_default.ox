def invalid_q31_sos_df2_fixed_default(
    input: tensor[q31],
    coefficients: tensor[q31,1,5],
    scales: tensor[q31,1],
    state: tensor[q31,1,2])
    -> (tensor[q31], tensor[q31,1,2]):
  return sos_df2_fixed(input, coefficients, scales, state)
