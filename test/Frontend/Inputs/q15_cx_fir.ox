def q15_cx_fir(input: tensor[complex_q15, 10], coefficients: tensor[complex_q15, 4])
    -> tensor[complex_q15, 7]:
  return cx_fir_filter(input, coefficients, boundary=valid,
                       accumulator=exact[32, saturate],
                       rounding=toward_negative,
                       overflow=saturate)
