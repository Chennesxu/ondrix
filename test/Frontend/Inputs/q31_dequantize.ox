def q31_dequantize(x: tensor[q31,64]) -> tensor[f32,64]:
  return dequantize(x, to=f32)
