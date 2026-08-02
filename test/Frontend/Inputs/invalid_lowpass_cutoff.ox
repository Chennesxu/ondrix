def invalid_lowpass_cutoff(signal: tensor[q15,72]) -> tensor[q15,33]:
  taps = lowpass(taps=9, cutoff=[1,2])
  filtered = fir_filter(signal, taps, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
  spectrum = rfft(filtered)
  return magnitude(spectrum)
