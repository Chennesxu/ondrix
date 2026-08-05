def invalid_window_taps(signal: tensor[q15,16]) -> tensor[q15,16]:
  taps = blackman(taps=1)
  filtered = fir_filter(signal, taps, boundary=valid)
  return abs(filtered)
