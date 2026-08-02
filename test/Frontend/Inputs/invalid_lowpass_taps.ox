def invalid_lowpass_taps(signal: tensor[q15,72]) -> tensor[q15,33]:
  taps = lowpass(taps=8, cutoff=[1,4])
  filtered = fir_filter(signal, taps, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
  spectrum = rfft(filtered)
  return magnitude(spectrum)
