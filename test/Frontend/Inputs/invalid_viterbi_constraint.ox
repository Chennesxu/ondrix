def viterbi_k9(symbols: tensor[q15, 512]) -> tensor[u8, 32]:
  return viterbi_decode(symbols, constraint_length=9, polynomials=[0o171, 0o133])
