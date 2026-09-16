def invalid_dct_raw_high_rounding(input: tensor[q31,8]) -> tensor[q31,8]:
  return dct(input, product=raw_high, product_rounding=nearest_even)
