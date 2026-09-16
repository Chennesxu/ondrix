def q31_dct64_raw_high(input: tensor[q31,64]) -> tensor[q31,64]:
  return dct(input, product=raw_high, rounding=toward_negative)
