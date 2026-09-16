def invalid_dct_raw_high_q15(input: tensor[q15,8]) -> tensor[q15,8]:
  return dct(input, product=raw_high)
