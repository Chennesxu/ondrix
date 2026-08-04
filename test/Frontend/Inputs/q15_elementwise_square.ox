def q15_elementwise_square(x: tensor[q15,16]) -> tensor[q15,16]:
  return mult(x, x)
