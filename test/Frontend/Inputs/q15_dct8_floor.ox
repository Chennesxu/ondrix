def q15_dct8_floor(x: tensor[q15,8]) -> tensor[q15,8]:
  return dct(x, rounding=toward_negative)
