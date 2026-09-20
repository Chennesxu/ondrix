def q31_cx_fir(input: tensor[complex_q31, 10], coefficients: tensor[complex_q31, 4])
    -> tensor[complex_q31, 7]:
  return cx_fir_filter(input, coefficients, boundary=valid,
                       accumulator=exact[64, saturate],
                       rounding=toward_negative,
                       overflow=saturate)
