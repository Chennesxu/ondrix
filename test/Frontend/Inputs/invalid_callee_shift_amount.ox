def bad(x: tensor[q15,16]) -> tensor[q15,16]:
  return shift(x, amount=-20)

def invalid_callee_shift_amount(x: tensor[q15,16]) -> tensor[q15,16]:
  return bad(x)
