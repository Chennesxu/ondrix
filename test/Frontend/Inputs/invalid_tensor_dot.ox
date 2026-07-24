def invalid_tensor_dot(lhs: tensor[q15], rhs: tensor[q15]) -> q15:
  return dot(lhs, rhs, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
