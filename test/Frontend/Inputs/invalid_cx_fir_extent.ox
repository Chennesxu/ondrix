def invalid_cx_fir_extent(input: tensor[complex_q15, 10], coefficients: tensor[complex_q15, 4])
    -> tensor[complex_q15, 10]:
  return cx_fir_filter(input, coefficients, boundary=valid,
                       accumulator=exact[32, saturate],
                       rounding=toward_negative,
                       overflow=saturate)
