def q31_dct8_raw_high(input: tensor[q31,8]) -> tensor[q31,8]:
  return dct(input, product=raw_high)
