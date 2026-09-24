def viterbi_unbounded(symbols: tensor[q15, 512]) -> tensor[u8, 32]:
  return viterbi_decode(symbols, constraint_length=7, polynomials=[0o171, 0o133], symbol_bound=32768)
