def window(x: tensor[q15,71], c: tensor[q15,8]) -> tensor[q15,64]:
  return fir_filter(x, c, boundary=valid)

def q15_call_spectrum(x: tensor[q15,71], c: tensor[q15,8]) -> tensor[q15,33]:
  return magnitude(rfft(window(x, c)))
