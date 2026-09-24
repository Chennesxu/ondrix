def viterbi_bytes(symbols: tensor[q15, 512]) -> tensor[u8, 64]:
  return viterbi_decode(symbols, constraint_length=7, polynomials=[0o171, 0o133])
