def invalid_quantize_operand(x: tensor[q15,64]) -> tensor[q31,64]:
  return quantize(x, to=q31)
