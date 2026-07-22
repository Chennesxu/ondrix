kernel f32_fir_fma(window: buffer[f32], coefficients: buffer[f32]) -> f32:
  return fir(window, coefficients, contract=fma)
