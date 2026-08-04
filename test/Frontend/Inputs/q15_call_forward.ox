def prefilter(x: tensor[q15,64], c: tensor[q15,8]) -> tensor[q15,57]:
  return fir_filter(x, c, boundary=valid,
                    accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)

def q15_call_forward(x: tensor[q15,64], c: tensor[q15,8]) -> tensor[q15,57]:
  return prefilter(x, c)
