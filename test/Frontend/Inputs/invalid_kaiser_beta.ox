def invalid_kaiser_beta(signal: tensor[q15,16]) -> tensor[q15,12]:
  taps = kaiser(taps=5, beta=[51,1])
  filtered = fir_filter(signal, taps, boundary=valid)
  return abs(filtered)
