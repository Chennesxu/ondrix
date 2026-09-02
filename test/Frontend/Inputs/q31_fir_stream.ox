def q31_fir_stream(
    input: tensor[q31],
    coefficients: tensor[q31,3],
    state: tensor[q31,2])
    -> (tensor[q31], tensor[q31,2]):
  return fir_stream(input, coefficients, state,
                    accumulator=exact[64,saturate],
                    rounding=nearest_even, overflow=saturate)
