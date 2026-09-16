def invalid_literal_call_operand(x: tensor[q15,16]) -> tensor[q15,16]:
  return add(x, 3)
