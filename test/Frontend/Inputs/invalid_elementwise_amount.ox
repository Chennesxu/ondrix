def invalid_elementwise_amount(x: tensor[q15,32]) -> tensor[q15,32]:
  return shift(x, amount=16)
