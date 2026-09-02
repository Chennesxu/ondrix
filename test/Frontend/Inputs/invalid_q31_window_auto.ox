def invalid_q31_window_auto(signal: tensor[q31,40]) -> tensor[q31,32]:
  taps = hamming(taps=9)
  filtered = fir_filter(signal, taps, boundary=valid)
  return abs(filtered)
