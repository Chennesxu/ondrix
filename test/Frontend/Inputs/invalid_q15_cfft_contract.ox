def invalid_q15_cfft_contract(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return cfft(input, contract = fma)
