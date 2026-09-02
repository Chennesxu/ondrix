def invalid_q31_fir_stream_default(
    input: tensor[q31],
    coefficients: tensor[q31,3],
    state: tensor[q31,2])
    -> (tensor[q31], tensor[q31,2]):
  return fir_stream(input, coefficients, state)
