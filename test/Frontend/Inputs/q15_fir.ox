kernel q15_fir(window: buffer[q15], coefficients: buffer[q15]) -> q15:
  return fir(window, coefficients,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
