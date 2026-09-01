def invalid_dct_product_boundary(x: tensor[q15,8]) -> tensor[q15,8]:
  return dct(x, product_rounding=toward_negative)
