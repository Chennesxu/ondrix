def q15_window_spectrum(signal: tensor[q15,40]) -> tensor[q15,32]:
  taps = hamming(taps=9)
  filtered = fir_filter(signal, taps, boundary=valid)
  return abs(filtered)
