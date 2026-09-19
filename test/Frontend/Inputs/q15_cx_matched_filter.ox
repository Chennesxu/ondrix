def q15_cx_matched_filter(input: tensor[complex_q15, 10], reference: tensor[complex_q15, 4])
    -> tensor[complex_q15, 7]:
  return cx_fir_filter(input, reference, conjugate=true, boundary=valid,
                       accumulator=exact[40, saturate],
                       rounding=nearest_ties_positive,
                       overflow=saturate)
