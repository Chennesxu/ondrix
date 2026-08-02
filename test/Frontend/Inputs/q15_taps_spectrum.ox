def q15_taps_spectrum(signal: tensor[q15,36], taps: tensor[q15,5]) -> tensor[q15,17]:
  filtered = fir_filter(signal, taps, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
  spectrum = rfft(filtered)
  return magnitude(spectrum)
