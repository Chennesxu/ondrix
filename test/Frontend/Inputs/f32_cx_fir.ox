def f32_cx_fir(signal: tensor[complex_f32, 12], taps: tensor[complex_f32, 4])
    -> tensor[complex_f32, 9]:
  return cx_fir_filter(signal, taps, conjugate=true, boundary=valid, contract=fma)
