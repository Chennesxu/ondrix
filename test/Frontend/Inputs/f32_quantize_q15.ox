def f32_quantize_q15(x: tensor[f32,64]) -> tensor[q15,64]:
  return quantize(x, to=q15)
