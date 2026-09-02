def q31_window_spectrum(signal: tensor[q31,40]) -> tensor[q31,32]:
  taps = hamming(taps=9)
  filtered = fir_filter(signal, taps, boundary=valid,
                        accumulator=exact[64,saturate],
                        rounding=nearest_even, overflow=saturate)
  return abs(filtered)
