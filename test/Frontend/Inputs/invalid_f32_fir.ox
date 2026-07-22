kernel f32_fir(window: buffer[f32], coefficients: buffer[f32]) -> f32:
  return fir(window, coefficients, contract=fma)
