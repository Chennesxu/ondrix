def f32_quantize_chain(x: tensor[f32,64], y: tensor[q15,64]) -> tensor[f32,64]:
  return dequantize(quantize(x, to=q15) + y, to=f32)
