def q15_conversion_chain(x: tensor[q15,32], y: tensor[q31,32]) -> tensor[q15,32]:
  return narrow(widen(x, to=q31) * y + widen(x / 2, to=q31), to=q15, rounding=nearest_even)
