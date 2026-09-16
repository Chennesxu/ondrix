def f32_quantize_q15_even(x: tensor[f32,64]) -> tensor[q15,64]:
  return quantize(x, to=q15, rounding=nearest_even, overflow=saturate)
