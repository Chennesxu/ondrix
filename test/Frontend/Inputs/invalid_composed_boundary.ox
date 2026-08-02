def invalid_composed_boundary(signal: tensor[q15,72]) -> tensor[q15,33]:
  taps = lowpass(taps=9, cutoff=[1,4])
  filtered = fir_filter(signal, taps, boundary=full, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
  spectrum = rfft(filtered)
  return magnitude(spectrum)
