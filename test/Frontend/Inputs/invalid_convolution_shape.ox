kernel invalid_convolution_shape(input: tensor[q15,6], kernel: tensor[q15,3]) -> tensor[q15,5]:
  return convolution(input, kernel, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
