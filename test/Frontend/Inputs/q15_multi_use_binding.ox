def q15_multi_use_binding(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  t = mult(x, y, rounding=nearest_even, overflow=saturate)
  square = mult(t, t, rounding=nearest_even, overflow=saturate)
  return add(shift(square, amount=-1, rounding=nearest_even, overflow=saturate),
             t, overflow=saturate)
