def invalid_elementwise_q31_amount(x: tensor[q31,32]) -> tensor[q31,32]:
  return shift(x, amount=32)
