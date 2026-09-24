def viterbi_k5_r3(symbols: tensor[q15, 192]) -> tensor[u8, 8]:
  return viterbi_decode(symbols, constraint_length=5, polynomials=[21, 27, 31])
