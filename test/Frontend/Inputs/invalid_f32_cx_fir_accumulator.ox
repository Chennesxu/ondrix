def invalid_f32_cx_fir_accumulator(signal: tensor[complex_f32, 12], taps: tensor[complex_f32, 4])
    -> tensor[complex_f32, 9]:
  return cx_fir_filter(signal, taps, boundary=valid,
                       accumulator=exact[64, saturate],
                       rounding=toward_negative,
                       overflow=saturate)
