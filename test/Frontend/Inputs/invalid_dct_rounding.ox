def q15_dct8_bad(x: tensor[q15,8]) -> tensor[q15,8]:
  return dct(x, rounding=toward_zero)
