def viterbi_nested(symbols: tensor[q15, 512]) -> tensor[q15, 32]:
  return abs(viterbi_decode(symbols, constraint_length=7, polynomials=[0o171, 0o133]))
