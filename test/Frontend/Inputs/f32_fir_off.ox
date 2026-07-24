def f32_fir_off(window: buffer[f32], coefficients: buffer[f32]) -> f32:
  return fir(window, coefficients, contract=off)
