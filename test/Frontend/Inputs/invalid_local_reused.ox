def invalid_local_reused(signal: tensor[q15,20], window: tensor[q15,5]) -> tensor[q15,9]:
  filtered = fir_filter(signal, window, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
  spectrum = rfft(filtered)
  return magnitude(filtered)
