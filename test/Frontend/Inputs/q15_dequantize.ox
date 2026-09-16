def q15_dequantize(x: tensor[q15,64]) -> tensor[f32,64]:
  return dequantize(x, to=f32)
