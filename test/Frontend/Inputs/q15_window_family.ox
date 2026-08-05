def q15_window_family(signal: tensor[q15,20]) -> tensor[q15,16]:
  hann_taps = hann(taps=5)
  blackman_taps = blackman(taps=5)
  kaiser_taps = kaiser(taps=5, beta=[17,2])
  hann_out = fir_filter(signal, hann_taps, boundary=valid)
  blackman_out = fir_filter(signal, blackman_taps, boundary=valid)
  kaiser_out = fir_filter(signal, kaiser_taps, boundary=valid)
  return add(sub(hann_out, blackman_out), kaiser_out)
