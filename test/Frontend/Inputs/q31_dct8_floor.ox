def q31_dct8_floor(x: tensor[q31,8]) -> tensor[q31,8]:
  return dct(x, product_rounding=toward_negative, rounding=toward_negative)
