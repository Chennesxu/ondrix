def viterbi_q31(symbols: tensor[q31, 512]) -> tensor[u8, 32]:
  return viterbi_decode(symbols, constraint_length=7, polynomials=[0o171, 0o133])
