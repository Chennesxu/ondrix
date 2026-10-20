#include "ondrix/Frontend/OxFrontend.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace ondrix::frontend {
namespace {

struct SourcePosition {
  size_t offset = 0;
  unsigned line = 1;
  unsigned column = 1;
};

class Diagnostics {
public:
  Diagnostics(llvm::StringRef sourceName, llvm::StringRef source, llvm::raw_ostream &output)
      : sourceName(sourceName), source(source), output(output) {}

  void error(SourcePosition position, const llvm::Twine &message) {
    hadError = true;
    output << sourceName << ':' << position.line << ':' << position.column << ": error: " << message
           << '\n';

    size_t lineStart = position.offset;
    while (lineStart > 0 && source[lineStart - 1] != '\n')
      --lineStart;
    size_t lineEnd = source.find('\n', position.offset);
    if (lineEnd == llvm::StringRef::npos)
      lineEnd = source.size();
    llvm::StringRef line = source.slice(lineStart, lineEnd);
    output << line << '\n';
    for (unsigned column = 1; column < position.column; ++column)
      output << ' ';
    output << "^\n";
  }

  bool failed() const { return hadError; }

private:
  llvm::StringRef sourceName;
  llvm::StringRef source;
  llvm::raw_ostream &output;
  bool hadError = false;
};

enum class TokenKind {
  Eof,
  Identifier,
  Integer,
  Float,
  EqualEqual,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  LeftParen,
  RightParen,
  LeftBracket,
  RightBracket,
  Comma,
  Colon,
  Equal,
  Minus,
  Plus,
  Star,
  Slash,
  Arrow,
  Invalid,
};

struct Token {
  TokenKind kind = TokenKind::Invalid;
  llvm::StringRef spelling;
  SourcePosition position;
};

class Lexer {
public:
  Lexer(llvm::StringRef source, Diagnostics &diagnostics)
      : source(source), diagnostics(diagnostics) {}

  Token next() {
    skipTrivia();
    SourcePosition start{offset, line, column};
    if (offset == source.size())
      return {TokenKind::Eof, {}, start};

    char current = source[offset];
    if (isIdentifierStart(current)) {
      size_t begin = offset;
      advance();
      while (offset < source.size() && isIdentifierContinue(source[offset]))
        advance();
      return {TokenKind::Identifier, source.slice(begin, offset), start};
    }
    if (std::isdigit(static_cast<unsigned char>(current))) {
      size_t begin = offset;
      // `0o` then octal digits, the notation generator polynomials are read in.
      auto isOctal = [](char c) { return c >= '0' && c <= '7'; };
      if (current == '0' && offset + 2 < source.size() && source[offset + 1] == 'o' &&
          isOctal(source[offset + 2])) {
        advance();
        advance();
        while (offset < source.size() && isOctal(source[offset]))
          advance();
        return {TokenKind::Integer, source.slice(begin, offset), start};
      }
      do {
        advance();
      } while (offset < source.size() && std::isdigit(static_cast<unsigned char>(source[offset])));
      bool floating = false;
      if (offset < source.size() && source[offset] == '.') {
        floating = true;
        advance();
        while (offset < source.size() && std::isdigit(static_cast<unsigned char>(source[offset])))
          advance();
      }
      if (offset < source.size() && (source[offset] == 'e' || source[offset] == 'E')) {
        floating = true;
        advance();
        if (offset < source.size() && (source[offset] == '+' || source[offset] == '-'))
          advance();
        size_t exponentStart = offset;
        while (offset < source.size() && std::isdigit(static_cast<unsigned char>(source[offset])))
          advance();
        if (offset == exponentStart)
          diagnostics.error(start, "invalid f32 literal: exponent requires digits");
      }
      return {floating ? TokenKind::Float : TokenKind::Integer, source.slice(begin, offset), start};
    }
    if (current == '=' || current == '!' || current == '<' || current == '>') {
      size_t begin = offset;
      advance();
      bool equal = offset < source.size() && source[offset] == '=';
      if (equal)
        advance();
      TokenKind kind = current == '='   ? (equal ? TokenKind::EqualEqual : TokenKind::Equal)
                       : current == '!' ? (equal ? TokenKind::NotEqual : TokenKind::Invalid)
                       : current == '<' ? (equal ? TokenKind::LessEqual : TokenKind::Less)
                                        : (equal ? TokenKind::GreaterEqual : TokenKind::Greater);
      return {kind, source.slice(begin, offset), start};
    }
    if (current == '-') {
      advance();
      if (offset < source.size() && source[offset] == '>') {
        advance();
        return {TokenKind::Arrow, "->", start};
      }
      return {TokenKind::Minus, "-", start};
    }

    advance();
    switch (current) {
    case '(':
      return {TokenKind::LeftParen, "(", start};
    case ')':
      return {TokenKind::RightParen, ")", start};
    case '[':
      return {TokenKind::LeftBracket, "[", start};
    case ']':
      return {TokenKind::RightBracket, "]", start};
    case ',':
      return {TokenKind::Comma, ",", start};
    case ':':
      return {TokenKind::Colon, ":", start};
    case '=':
      return {TokenKind::Equal, "=", start};
    case '+':
      return {TokenKind::Plus, "+", start};
    case '*':
      return {TokenKind::Star, "*", start};
    case '/':
      return {TokenKind::Slash, "/", start};
    default:
      diagnostics.error(start,
                        llvm::Twine("unexpected character '") + llvm::StringRef(&current, 1) + "'");
      return {TokenKind::Invalid, source.slice(start.offset, offset), start};
    }
  }

private:
  static bool isIdentifierStart(char value) {
    return value == '_' || std::isalpha(static_cast<unsigned char>(value));
  }

  static bool isIdentifierContinue(char value) {
    return value == '_' || std::isalnum(static_cast<unsigned char>(value));
  }

  void advance() {
    if (source[offset++] == '\n') {
      ++line;
      column = 1;
    } else {
      ++column;
    }
  }

  void skipTrivia() {
    while (offset < source.size()) {
      if (std::isspace(static_cast<unsigned char>(source[offset]))) {
        advance();
        continue;
      }
      if (source[offset] == '#') {
        while (offset < source.size() && source[offset] != '\n')
          advance();
        continue;
      }
      break;
    }
  }

  llvm::StringRef source;
  Diagnostics &diagnostics;
  size_t offset = 0;
  unsigned line = 1;
  unsigned column = 1;
};

enum class SourceType {
  Q15,
  Q31,
  F32,
  ComplexQ15,
  ComplexQ31,
  ComplexF32,
  U8,
  I32,
  I64,
  Bool,
  Index
};

// The value of an integer token, decimal or `0o` octal; true on failure, as
// StringRef::getAsInteger.
static bool getIntegerTokenValue(llvm::StringRef spelling, uint64_t &value) {
  if (spelling.consume_front("0o"))
    return spelling.getAsInteger(8, value);
  return spelling.getAsInteger(10, value);
}

enum class ContainerKind { Scalar, Buffer, Tensor, Constexpr };

enum class ReductionKind {
  Primitive,
  Dot,
  CxDot,
  CxFirFilter,
  Fir,
  FirFilter,
  FirDecimate,
  FirInterpolate,
  FirStream,
  SosDf2Fixed,
  Convolution,
  Correlation,
  Butterfly,
  Cfft,
  Icfft,
  Rfft,
  Irfft,
  Magnitude,
  Power,
  Phase,
  Dct,
  MovingAverage,
  Gain,
  Rms,
  Sine,
  Cosine,
  Matmul,
  Lms,
  Goertzel,
  SosTdf2,
  Lowpass,
  Hamming,
  Hann,
  Blackman,
  Kaiser,
  CicDecimate,
  Add,
  Sub,
  Mult,
  Abs,
  Negate,
  Offset,
  Shift,
  Div,
  Ratio,
  Widen,
  Narrow,
  Quantize,
  Dequantize,
  Log2,
  Exp2,
  ViterbiDecode
};

// The elementwise family: every member is one exact integer expression plus
// one declared boundary, and each has a statically known result shape, which
// is what lets them appear as operands of one another.
static bool isElementwiseKind(ReductionKind kind) {
  return kind == ReductionKind::Add || kind == ReductionKind::Sub || kind == ReductionKind::Mult ||
         kind == ReductionKind::Abs || kind == ReductionKind::Negate ||
         kind == ReductionKind::Offset || kind == ReductionKind::Shift ||
         kind == ReductionKind::Div || kind == ReductionKind::Ratio;
}

static bool isBinaryElementwiseKind(ReductionKind kind) {
  return kind == ReductionKind::Add || kind == ReductionKind::Sub || kind == ReductionKind::Mult ||
         kind == ReductionKind::Ratio;
}

// The four spellings of the one conversion operation: the name fixes the
// direction, so a target on the wrong side is refused, never reinterpreted.
static bool isConversionKind(ReductionKind kind) {
  return kind == ReductionKind::Widen || kind == ReductionKind::Narrow ||
         kind == ReductionKind::Quantize || kind == ReductionKind::Dequantize;
}

// The two that carry a boundary and therefore take rounding and overflow.
static bool isLossyConversionKind(ReductionKind kind) {
  return kind == ReductionKind::Narrow || kind == ReductionKind::Quantize;
}

// The compile-time coefficient designs. None has a runtime form, so each is
// legal only in the coefficient slot of a composed fir_filter.
static bool isWindowDesignKind(ReductionKind kind) {
  return kind == ReductionKind::Hamming || kind == ReductionKind::Hann ||
         kind == ReductionKind::Blackman || kind == ReductionKind::Kaiser;
}

static bool isDesignKind(ReductionKind kind) {
  return kind == ReductionKind::Lowpass || isWindowDesignKind(kind);
}

static llvm::StringRef describeDesignKind(ReductionKind kind) {
  switch (kind) {
  case ReductionKind::Hamming:
    return "hamming";
  case ReductionKind::Hann:
    return "hann";
  case ReductionKind::Blackman:
    return "blackman";
  case ReductionKind::Kaiser:
    return "kaiser";
  default:
    return "lowpass";
  }
}

static bool isCfftKind(ReductionKind kind) {
  return kind == ReductionKind::Cfft || kind == ReductionKind::Icfft;
}

static bool isFftKind(ReductionKind kind) {
  return isCfftKind(kind) || kind == ReductionKind::Rfft || kind == ReductionKind::Irfft;
}

static bool isFftComposableKind(ReductionKind kind) {
  return isFftKind(kind) || kind == ReductionKind::Magnitude || kind == ReductionKind::Power ||
         kind == ReductionKind::Phase;
}

// One operand whose result shape follows from it and the call's attributes.
static bool isUnaryComposableKind(ReductionKind kind) {
  return isFftComposableKind(kind) || kind == ReductionKind::ViterbiDecode;
}

// The set whose members may hold one another as operands. Its checker
// carries an element type and extent from operand to result, so a member has
// to have a statically derivable result shape.
static bool isComposableKind(ReductionKind kind) {
  return isUnaryComposableKind(kind) || isElementwiseKind(kind) || isConversionKind(kind);
}

// A fixed gain is one requantization with a static shape, so it nests and
// leads a kernel like the elementwise family; the f32 gain keeps its own path.
static bool isComposableKernel(ReductionKind kind, SourceType resultType) {
  return kind == ReductionKind::Primitive || isComposableKind(kind) ||
         (kind == ReductionKind::Gain && resultType != SourceType::F32);
}

static bool isUnaryTensorKind(ReductionKind kind) {
  return kind == ReductionKind::Dct || kind == ReductionKind::MovingAverage ||
         kind == ReductionKind::Gain || kind == ReductionKind::Rms || kind == ReductionKind::Sine ||
         kind == ReductionKind::Cosine || kind == ReductionKind::CicDecimate ||
         kind == ReductionKind::Log2 || kind == ReductionKind::Exp2 ||
         kind == ReductionKind::Goertzel;
}

// The reductions with a scalar result, which an ordinary expression can hold.
static bool isScalarReductionKind(ReductionKind kind) {
  return kind == ReductionKind::Dot || kind == ReductionKind::Fir || kind == ReductionKind::Rms;
}

static bool isUnaryKind(ReductionKind kind) {
  return isUnaryComposableKind(kind) || isUnaryTensorKind(kind) || isConversionKind(kind) ||
         (isElementwiseKind(kind) && !isBinaryElementwiseKind(kind));
}

struct ParameterAst {
  std::string name;
  SourceType type;
  std::vector<std::optional<int64_t>> shape;
  ContainerKind container = ContainerKind::Buffer;
  std::vector<int64_t> constantValues;
  SourcePosition position;

  bool isBuffer() const { return container == ContainerKind::Buffer; }
  bool isScalar() const { return container == ContainerKind::Scalar; }
  bool isTensor() const { return container == ContainerKind::Tensor; }
  bool isConstexpr() const { return container == ContainerKind::Constexpr; }
};

struct BuiltinCallAst;
struct CheckedKernel;

struct ExpressionAst {
  std::string parameter;
  std::unique_ptr<BuiltinCallAst> call;
  SourcePosition position;

  ExpressionAst(std::string parameter, SourcePosition position)
      : parameter(std::move(parameter)), position(position) {}
  explicit ExpressionAst(BuiltinCallAst call);
  ~ExpressionAst();
  ExpressionAst(ExpressionAst &&) noexcept;
  ExpressionAst &operator=(ExpressionAst &&) noexcept;
  // A deep copy, so instantiating a callee copies its call tree WHOLE. The
  // one field-by-field copier this replaced silently dropped the two fields
  // added after it was written; a compiler-generated copy cannot repeat that.
  ExpressionAst(const ExpressionAst &);
  ExpressionAst &operator=(const ExpressionAst &) = delete;

  bool isParameterReference() const { return !call; }
};

enum class PrimitiveKind {
  Identity,
  Literal,
  Add,
  Sub,
  Mul,
  Div,
  Pos,
  Neg,
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
  Not,
  And,
  Or,
  // Statement sequences: operand i is assigned to bindingNames[i], or is a
  // nested Loop under an empty name. A Scope's last operand is its result. A
  // Loop's operands are start, stop, then its body; bindingNames[0] names the
  // induction variable.
  Scope,
  Loop,
  Length,
  Element,
  Typed
};

// Operators parse as primitives and map back to the same-spelled builtin on
// fixed operands; infix `/` is `ratio`.
static constexpr std::pair<ReductionKind, PrimitiveKind> kOperatorKinds[] = {
    {ReductionKind::Add, PrimitiveKind::Add},    {ReductionKind::Sub, PrimitiveKind::Sub},
    {ReductionKind::Mult, PrimitiveKind::Mul},   {ReductionKind::Ratio, PrimitiveKind::Div},
    {ReductionKind::Negate, PrimitiveKind::Neg},
};

static std::optional<PrimitiveKind> getOperatorPrimitive(ReductionKind kind) {
  for (auto [builtin, primitive] : kOperatorKinds)
    if (builtin == kind)
      return primitive;
  return std::nullopt;
}

static std::optional<ReductionKind> getOperatorBuiltin(PrimitiveKind kind) {
  for (auto [builtin, primitive] : kOperatorKinds)
    if (primitive == kind)
      return builtin;
  return std::nullopt;
}

// The builtin call spellings that also accept ordinary operands.
static std::optional<PrimitiveKind> getOrdinaryArithmetic(ReductionKind kind) {
  return kind == ReductionKind::Ratio ? std::nullopt : getOperatorPrimitive(kind);
}

static bool isComparison(PrimitiveKind kind) {
  return llvm::is_contained({PrimitiveKind::Eq, PrimitiveKind::Ne, PrimitiveKind::Lt,
                             PrimitiveKind::Le, PrimitiveKind::Gt, PrimitiveKind::Ge},
                            kind);
}

struct BuiltinCallAst {
  ReductionKind kind;
  PrimitiveKind primitive = PrimitiveKind::Identity;
  std::string spelling;
  std::vector<std::string> bindingNames;
  SourceType expressionType = SourceType::I32;
  int64_t expressionExtent = 0;
  int64_t declaredExtent = 0;
  std::optional<llvm::APInt> integerConstant;
  std::vector<ExpressionAst> operands;
  uint64_t accumulatorWidth = 0;
  bool accumulatorAuto = false;
  std::string updateOverflow;
  std::string rounding;
  std::string inputRounding;
  std::string destinationOverflow;
  std::string stateRounding;
  std::string stateOverflow;
  std::string fpContract;
  std::string boundary;
  std::string turn;
  int64_t factor = 0;
  int64_t window = 0;
  int64_t gain = 0;
  int64_t stepSize = 0;
  int64_t fpConstantDen = 0;
  float fpConstant = 0.0f;
  int64_t taps = 0;
  int64_t cutoffNum = 0;
  int64_t cutoffDen = 0;
  int64_t betaNum = 0;
  int64_t betaDen = 0;
  int64_t bin = 0;
  int64_t stages = 0;
  int64_t rate = 0;
  int64_t delay = 0;
  int64_t bias = 0;
  int64_t amount = 0;
  int64_t divisor = 0;
  std::string nonpositive;
  std::string product;
  bool conjugate = false;
  bool normalized = false;
  int64_t epsilon = 0;
  int64_t gainBound = 0;
  int64_t symbolBound = 0;
  int64_t constraintLength = 0;
  std::vector<int64_t> polynomials;
  SourceType target = SourceType::Q15;
  bool literal = false;
  int64_t step = 1;
  std::vector<std::string> carriedNames;
  bool inBounds = false;
  // A reduction inside an ordinary expression, checked as a kernel of its own.
  std::shared_ptr<const CheckedKernel> library;
  SourcePosition position;
};

enum class SignFact { Unknown, NonNegative, Positive };

// What the structure of a composed expression proves about its sign: a
// magnitude or a square of one operand is non-negative, a positive constant
// added under saturation makes it positive, and no operand of unknown sign
// becomes known. This is what admits an unspelled divisor policy.
static SignFact signFact(const ExpressionAst &expression) {
  if (expression.isParameterReference())
    return SignFact::Unknown;
  const BuiltinCallAst &call = *expression.call;
  bool saturating = call.destinationOverflow != "wrap";
  auto known = [](SignFact fact) { return fact != SignFact::Unknown; };
  switch (call.kind) {
  case ReductionKind::Abs:
    return saturating ? SignFact::NonNegative : SignFact::Unknown;
  case ReductionKind::Mult: {
    const ExpressionAst &lhs = call.operands[0], &rhs = call.operands[1];
    bool square =
        lhs.isParameterReference() && rhs.isParameterReference() && lhs.parameter == rhs.parameter;
    return saturating && square ? SignFact::NonNegative : SignFact::Unknown;
  }
  case ReductionKind::Offset: {
    SignFact input = signFact(call.operands[0]);
    if (!saturating || !known(input) || call.bias < 0)
      return SignFact::Unknown;
    return call.bias > 0 ? SignFact::Positive : input;
  }
  case ReductionKind::Add: {
    SignFact lhs = signFact(call.operands[0]), rhs = signFact(call.operands[1]);
    if (!saturating || !known(lhs) || !known(rhs))
      return SignFact::Unknown;
    return lhs == SignFact::Positive || rhs == SignFact::Positive ? SignFact::Positive
                                                                  : SignFact::NonNegative;
  }
  case ReductionKind::Shift: {
    SignFact input = signFact(call.operands[0]);
    if (!saturating || !known(input))
      return SignFact::Unknown;
    return call.amount >= 0 ? input : SignFact::NonNegative;
  }
  case ReductionKind::Widen:
    return signFact(call.operands[0]);
  case ReductionKind::Div:
  case ReductionKind::Narrow:
    return known(signFact(call.operands[0])) ? SignFact::NonNegative : SignFact::Unknown;
  case ReductionKind::Gain:
    return known(signFact(call.operands[0])) && call.gain >= 0 ? SignFact::NonNegative
                                                               : SignFact::Unknown;
  default:
    return SignFact::Unknown;
  }
}

// Both operands bounded at 2^24 are exact in binary32, so dividing there
// gives the one correctly rounded quotient and no double-rounding argument is
// needed; 548055821/548055723 is the reachable pair that breaks the wider
// bound. APFloat rather than host arithmetic, so the result does not depend on
// the compiler's rounding environment. Contract: docs/frontend-language.md.
enum class RationalRefusal { None, Denominator, Magnitude };

std::pair<float, RationalRefusal> roundRationalToF32(int64_t numerator, int64_t denominator) {
  constexpr int64_t kExactInBinary32 = int64_t(1) << 24;
  if (denominator <= 0)
    return {0.0f, RationalRefusal::Denominator};
  if (denominator > kExactInBinary32 || numerator > kExactInBinary32 ||
      numerator < -kExactInBinary32)
    return {0.0f, RationalRefusal::Magnitude};
  llvm::APFloat quotient(static_cast<float>(numerator));
  llvm::APFloat divisor(static_cast<float>(denominator));
  quotient.divide(divisor, llvm::APFloat::rmNearestTiesToEven);
  return {quotient.convertToFloat(), RationalRefusal::None};
}

llvm::StringRef describeRationalRefusal(RationalRefusal refusal) {
  switch (refusal) {
  case RationalRefusal::Denominator:
    return "denominator must be positive";
  case RationalRefusal::Magnitude:
    return "numerator and denominator must not exceed 2^24, the bound that keeps both operands "
           "exact in the target format";
  case RationalRefusal::None:
    break;
  }
  llvm_unreachable("no refusal to describe");
}

ExpressionAst::ExpressionAst(BuiltinCallAst call)
    : call(std::make_unique<BuiltinCallAst>(std::move(call))), position(this->call->position) {}
ExpressionAst::~ExpressionAst() = default;
ExpressionAst::ExpressionAst(ExpressionAst &&) noexcept = default;
ExpressionAst::ExpressionAst(const ExpressionAst &other)
    : parameter(other.parameter),
      call(other.call ? std::make_unique<BuiltinCallAst>(*other.call) : nullptr),
      position(other.position) {}
ExpressionAst &ExpressionAst::operator=(ExpressionAst &&) noexcept = default;

struct Operand {
  std::optional<ExpressionAst> expression;
  int64_t literal = 0;
  SourcePosition position;
};

static std::optional<BuiltinCallAst> buildFixedBinary(ReductionKind kind, Operand lhs, Operand rhs,
                                                      SourcePosition position,
                                                      Diagnostics &diagnostics) {
  if (kind == ReductionKind::Ratio && !lhs.expression) {
    diagnostics.error(position, "a constant dividend is not an operation; a tensor divides "
                                "by a constant or by a tensor");
    return std::nullopt;
  }
  BuiltinCallAst call;
  call.kind = kind;
  call.position = position;
  if (lhs.expression && rhs.expression) {
    call.operands.push_back(std::move(*lhs.expression));
    call.operands.push_back(std::move(*rhs.expression));
    return call;
  }
  if (!lhs.expression && !rhs.expression) {
    diagnostics.error(position, "an operator needs a tensor operand on at least one side");
    return std::nullopt;
  }
  bool literalOnLeft = !lhs.expression;
  int64_t literal = literalOnLeft ? lhs.literal : rhs.literal;
  if (kind == ReductionKind::Sub) {
    if (literalOnLeft) {
      diagnostics.error(position, "a constant minus a tensor is negate then offset, two "
                                  "boundaries; spell offset(negate(x), bias=c)");
      return std::nullopt;
    }
    if (literal == std::numeric_limits<int64_t>::min()) {
      diagnostics.error(position, "integer literal is out of range");
      return std::nullopt;
    }
    literal = -literal;
  }
  call.operands.push_back(std::move(*(literalOnLeft ? rhs.expression : lhs.expression)));
  if (kind == ReductionKind::Ratio) {
    call.kind = ReductionKind::Div;
    call.divisor = literal;
  } else {
    call.kind = kind == ReductionKind::Mult ? ReductionKind::Gain : ReductionKind::Offset;
    (kind == ReductionKind::Mult ? call.gain : call.bias) = literal;
    call.literal = true;
  }
  return call;
}

// Instantiating a named function substitutes the caller's argument
// expressions for the callee's parameter references. Everything else — every
// declared contract in the callee's body — is copied unchanged, which is what
// makes the contract travel with the name.
static ExpressionAst instantiateExpression(const ExpressionAst &expression,
                                           const llvm::StringMap<const ExpressionAst *> &arguments,
                                           std::optional<SourcePosition> callSite);

// A null call site keeps the original positions, which is what checking a
// callee against its own signature needs.
static BuiltinCallAst instantiateCall(const BuiltinCallAst &call,
                                      const llvm::StringMap<const ExpressionAst *> &arguments,
                                      std::optional<SourcePosition> callSite) {
  BuiltinCallAst copy = call;
  copy.position = callSite.value_or(call.position);
  copy.operands.clear();
  for (const ExpressionAst &operand : call.operands)
    copy.operands.push_back(instantiateExpression(operand, arguments, callSite));
  return copy;
}

static ExpressionAst instantiateExpression(const ExpressionAst &expression,
                                           const llvm::StringMap<const ExpressionAst *> &arguments,
                                           std::optional<SourcePosition> callSite) {
  if (expression.isParameterReference()) {
    auto argument = arguments.find(expression.parameter);
    if (argument == arguments.end())
      return ExpressionAst(expression.parameter, callSite.value_or(expression.position));
    if (argument->second->isParameterReference())
      return ExpressionAst(argument->second->parameter,
                           callSite.value_or(argument->second->position));
    return ExpressionAst(instantiateCall(*argument->second->call, {}, callSite));
  }
  return ExpressionAst(instantiateCall(*expression.call, arguments, callSite));
}

struct ResultTypeAst {
  SourceType type;
  bool tensor = false;
  std::vector<std::optional<int64_t>> shape;
};

struct KernelAst {
  std::string name;
  std::vector<ParameterAst> parameters;
  std::vector<ResultTypeAst> results;
  BuiltinCallAst result;
  SourcePosition position;

  ResultTypeAst &primaryResult() { return results.front(); }
  const ResultTypeAst &primaryResult() const { return results.front(); }
};

static bool hasRank(llvm::ArrayRef<std::optional<int64_t>> shape, unsigned rank) {
  return shape.size() == rank;
}

static const std::optional<int64_t> &
getRankOneExtent(llvm::ArrayRef<std::optional<int64_t>> shape) {
  assert(hasRank(shape, 1));
  return shape.front();
}

class Parser {
public:
  Parser(Lexer &lexer, Diagnostics &diagnostics)
      : lexer(lexer), diagnostics(diagnostics), current(lexer.next()), next(lexer.next()) {}

  // A file declares one or more functions. The last is the kernel the module
  // exports; every earlier one is a named body a later function may call.
  std::optional<KernelAst> parse() {
    while (true) {
      bindings.clear();
      bindingsByName.clear();
      scopedOut.clear();
      scopedOutUse.reset();
      std::optional<KernelAst> function = parseFunction();
      if (!function)
        return std::nullopt;
      if (current.kind == TokenKind::Eof)
        return function;
      if (calleesByName.contains(function->name)) {
        diagnostics.error(function->position,
                          llvm::Twine("function '") + function->name + "' is already declared");
        return std::nullopt;
      }
      calleesByName[function->name] = callees.size();
      callees.push_back(std::move(*function));
    }
  }

  llvm::ArrayRef<KernelAst> getCallees() const { return callees; }

  std::optional<KernelAst> parseFunction() {
    if (!isIdentifier("def")) {
      if (current.kind == TokenKind::Identifier)
        diagnostics.error(current.position, llvm::Twine("unsupported top-level construct '") +
                                                current.spelling + "'; expected 'def'");
      else
        diagnostics.error(current.position, "expected 'def'");
      return std::nullopt;
    }

    KernelAst kernel;
    kernel.position = current.position;
    advance();
    auto name = parseIdentifier("expected function name");
    if (!name)
      return std::nullopt;
    kernel.name = name->spelling.str();

    if (!expect(TokenKind::LeftParen, "expected '(' after function name"))
      return std::nullopt;
    while (current.kind != TokenKind::RightParen) {
      std::optional<ParameterAst> parameter = parseParameter();
      if (!parameter)
        return std::nullopt;
      kernel.parameters.push_back(std::move(*parameter));
      if (current.kind != TokenKind::Comma)
        break;
      advance();
    }

    if (!expect(TokenKind::RightParen, "expected ')' after parameters") ||
        !expect(TokenKind::Arrow, "expected '->' after parameters"))
      return std::nullopt;
    if (current.kind == TokenKind::LeftParen) {
      advance();
      auto parseResult = [&](ResultTypeAst &result, llvm::StringRef ordinal) {
        if (isIdentifier("tensor")) {
          result.tensor = true;
          advance();
          return parseShapedType(result.type, result.shape, "tensor");
        }
        auto parsed = parseSourceType(llvm::Twine("expected ") + ordinal + " function result type");
        if (!parsed)
          return false;
        result.type = *parsed;
        return true;
      };
      ResultTypeAst first;
      if (!parseResult(first, "first") ||
          !expect(TokenKind::Comma, "expected ',' between function result types"))
        return std::nullopt;
      ResultTypeAst second;
      if (!parseResult(second, "second") ||
          !expect(TokenKind::RightParen, "expected ')' after function results"))
        return std::nullopt;
      if (first.type != second.type) {
        diagnostics.error(current.position, "multi-result functions require matching result types");
        return std::nullopt;
      }
      kernel.results.push_back(std::move(first));
      kernel.results.push_back(std::move(second));
    } else if (isIdentifier("tensor")) {
      ResultTypeAst result;
      result.tensor = true;
      advance();
      if (!parseShapedType(result.type, result.shape, "tensor"))
        return std::nullopt;
      kernel.results.push_back(std::move(result));
    } else {
      auto resultType = parseSourceType("expected function result type 'q15', 'q31', or 'f32'");
      if (!resultType)
        return std::nullopt;
      kernel.results.push_back(ResultTypeAst{*resultType, false, {}});
    }
    if (!expect(TokenKind::Colon, "expected ':' before function body"))
      return std::nullopt;

    // Local bindings before the single return statement: each names one
    // builtin call that a later statement reads at least once, and each read
    // instantiates the bound call — the checked kernel is the nested
    // expression tree direct nesting would produce.
    SourceType policyType = kernel.primaryResult().type;
    caller = &kernel;
    genericExpressions = policyType == SourceType::F32 || policyType == SourceType::I32 ||
                         policyType == SourceType::I64 || policyType == SourceType::Bool;
    BuiltinCallAst body = makePrimitive(PrimitiveKind::Scope, {}, current.position);
    while (isStatementStart())
      if (!parseStatement(body, policyType))
        return std::nullopt;
    if (isLoopStart()) {
      diagnostics.error(current.position,
                        "loops require an ordinary i32, i64, bool or f32 function result");
      return std::nullopt;
    }
    if (!expectIdentifier("return", "expected a single return statement"))
      return std::nullopt;
    std::optional<BuiltinCallAst> result = parseStatementExpression(policyType);
    if (!result)
      return std::nullopt;
    kernel.result = std::move(*result);
    if (genericExpressions && !body.operands.empty())
      kernel.result = bindLocals(std::move(body), std::move(kernel.result));
    caller = nullptr;
    if (current.kind != TokenKind::Eof && !isIdentifier("def")) {
      diagnostics.error(current.position,
                        "a function body is a single return statement; expected 'def' or "
                        "end of file");
      return std::nullopt;
    }
    if (scopedOutUse) {
      diagnostics.error(scopedOutUse->position, llvm::Twine("local '") + scopedOutUse->spelling +
                                                    "' is only visible inside its loop body");
      return std::nullopt;
    }
    for (const Binding &binding : bindings) {
      if (!binding.consumed) {
        diagnostics.error(binding.position, llvm::Twine("local '") + binding.sourceName +
                                                "' is never consumed by a later statement");
        return std::nullopt;
      }
    }
    return kernel;
  }

  // Without a requested policy type the call takes its first operand's, which
  // is how a call inside an ordinary expression spells its policies.
  std::optional<BuiltinCallAst> parseBuiltinCall(std::optional<SourceType> requestedPolicy) {
    if (current.kind == TokenKind::Identifier && next.kind == TokenKind::LeftParen &&
        calleesByName.contains(current.spelling))
      return parseCalleeInstantiation(requestedPolicy);
    if (caller && current.kind == TokenKind::Identifier && next.kind == TokenKind::LeftParen &&
        current.spelling == caller->name) {
      diagnostics.error(current.position,
                        llvm::Twine("function '") + current.spelling +
                            "' cannot call itself; a callee must be declared earlier in the file");
      return std::nullopt;
    }
    BuiltinCallAst call;
    if (!isIdentifier("dot") && !isIdentifier("fir") && !isIdentifier("fir_filter") &&
        !isIdentifier("fir_decimate") && !isIdentifier("fir_interpolate") &&
        !isIdentifier("fir_stream") && !isIdentifier("sos_df2_fixed") &&
        !isIdentifier("sos_tdf2") && !isIdentifier("goertzel") && !isIdentifier("hamming") &&
        !isIdentifier("hann") && !isIdentifier("blackman") && !isIdentifier("kaiser") &&
        !isIdentifier("convolution") && !isIdentifier("correlation") &&
        !isIdentifier("butterfly") && !isIdentifier("cfft") && !isIdentifier("icfft") &&
        !isIdentifier("rfft") && !isIdentifier("irfft") && !isIdentifier("magnitude") &&
        !isIdentifier("power") && !isIdentifier("phase") && !isIdentifier("dct") &&
        !isIdentifier("moving_average") && !isIdentifier("gain") && !isIdentifier("rms") &&
        !isIdentifier("sine") && !isIdentifier("cosine") && !isIdentifier("matmul") &&
        !isIdentifier("lms") && !isIdentifier("nlms") && !isIdentifier("lowpass") &&
        !isIdentifier("cic_decimate") && !isIdentifier("add") && !isIdentifier("sub") &&
        !isIdentifier("mult") && !isIdentifier("abs") && !isIdentifier("negate") &&
        !isIdentifier("offset") && !isIdentifier("shift") && !isIdentifier("div") &&
        !isIdentifier("ratio") && !isIdentifier("widen") && !isIdentifier("narrow") &&
        !isIdentifier("quantize") && !isIdentifier("dequantize") && !isIdentifier("log2") &&
        !isIdentifier("exp2") && !isIdentifier("cx_dot") && !isIdentifier("cx_fir_filter") &&
        !isIdentifier("viterbi_decode")) {
      diagnostics.error(
          current.position,
          "expected dot(...), cx_dot(...), fir(...), fir_filter(...), "
          "cx_fir_filter(...), fir_decimate(...), "
          "fir_interpolate(...), fir_stream(...), sos_df2_fixed(...), "
          "sos_tdf2(...), goertzel(...), "
          "convolution(...), correlation(...), butterfly(...), cfft(...), or "
          "icfft(...), rfft(...), irfft(...), magnitude(...), phase(...), dct(...), "
          "moving_average(...), gain(...), rms(...), power(...), sine(...), cosine(...), "
          "matmul(...), lms(...), nlms(...), cic_decimate(...), viterbi_decode(...), "
          "a lowpass/hamming/hann/"
          "blackman/kaiser design, or an "
          "elementwise add/sub/mult/abs/negate/offset/shift/div/ratio builtin "
          "expression, or a widen/narrow/quantize/dequantize conversion");
      return std::nullopt;
    }
    if (isIdentifier("dot"))
      call.kind = ReductionKind::Dot;
    else if (isIdentifier("cx_dot"))
      call.kind = ReductionKind::CxDot;
    else if (isIdentifier("cx_fir_filter"))
      call.kind = ReductionKind::CxFirFilter;
    else if (isIdentifier("fir"))
      call.kind = ReductionKind::Fir;
    else if (isIdentifier("fir_filter"))
      call.kind = ReductionKind::FirFilter;
    else if (isIdentifier("fir_decimate"))
      call.kind = ReductionKind::FirDecimate;
    else if (isIdentifier("fir_interpolate"))
      call.kind = ReductionKind::FirInterpolate;
    else if (isIdentifier("fir_stream"))
      call.kind = ReductionKind::FirStream;
    else if (isIdentifier("sos_df2_fixed"))
      call.kind = ReductionKind::SosDf2Fixed;
    else if (isIdentifier("sos_tdf2"))
      call.kind = ReductionKind::SosTdf2;
    else if (isIdentifier("goertzel"))
      call.kind = ReductionKind::Goertzel;
    else if (isIdentifier("viterbi_decode"))
      call.kind = ReductionKind::ViterbiDecode;
    else if (isIdentifier("hamming"))
      call.kind = ReductionKind::Hamming;
    else if (isIdentifier("hann"))
      call.kind = ReductionKind::Hann;
    else if (isIdentifier("blackman"))
      call.kind = ReductionKind::Blackman;
    else if (isIdentifier("kaiser"))
      call.kind = ReductionKind::Kaiser;
    else if (isIdentifier("convolution"))
      call.kind = ReductionKind::Convolution;
    else if (isIdentifier("correlation"))
      call.kind = ReductionKind::Correlation;
    else if (isIdentifier("butterfly"))
      call.kind = ReductionKind::Butterfly;
    else if (isIdentifier("cfft"))
      call.kind = ReductionKind::Cfft;
    else if (isIdentifier("icfft"))
      call.kind = ReductionKind::Icfft;
    else if (isIdentifier("rfft"))
      call.kind = ReductionKind::Rfft;
    else if (isIdentifier("irfft"))
      call.kind = ReductionKind::Irfft;
    else if (isIdentifier("magnitude"))
      call.kind = ReductionKind::Magnitude;
    else if (isIdentifier("power"))
      call.kind = ReductionKind::Power;
    else if (isIdentifier("phase"))
      call.kind = ReductionKind::Phase;
    else if (isIdentifier("dct"))
      call.kind = ReductionKind::Dct;
    else if (isIdentifier("moving_average"))
      call.kind = ReductionKind::MovingAverage;
    else if (isIdentifier("gain"))
      call.kind = ReductionKind::Gain;
    else if (isIdentifier("rms"))
      call.kind = ReductionKind::Rms;
    else if (isIdentifier("sine"))
      call.kind = ReductionKind::Sine;
    else if (isIdentifier("cosine"))
      call.kind = ReductionKind::Cosine;
    else if (isIdentifier("matmul"))
      call.kind = ReductionKind::Matmul;
    else if (isIdentifier("lms"))
      call.kind = ReductionKind::Lms;
    else if (isIdentifier("nlms")) {
      call.kind = ReductionKind::Lms;
      call.normalized = true;
    } else if (isIdentifier("cic_decimate"))
      call.kind = ReductionKind::CicDecimate;
    else if (isIdentifier("add"))
      call.kind = ReductionKind::Add;
    else if (isIdentifier("sub"))
      call.kind = ReductionKind::Sub;
    else if (isIdentifier("mult"))
      call.kind = ReductionKind::Mult;
    else if (isIdentifier("div"))
      call.kind = ReductionKind::Div;
    else if (isIdentifier("ratio"))
      call.kind = ReductionKind::Ratio;
    else if (isIdentifier("widen"))
      call.kind = ReductionKind::Widen;
    else if (isIdentifier("narrow"))
      call.kind = ReductionKind::Narrow;
    else if (isIdentifier("quantize"))
      call.kind = ReductionKind::Quantize;
    else if (isIdentifier("dequantize"))
      call.kind = ReductionKind::Dequantize;
    else if (isIdentifier("abs"))
      call.kind = ReductionKind::Abs;
    else if (isIdentifier("negate"))
      call.kind = ReductionKind::Negate;
    else if (isIdentifier("offset"))
      call.kind = ReductionKind::Offset;
    else if (isIdentifier("shift"))
      call.kind = ReductionKind::Shift;
    else if (isIdentifier("log2"))
      call.kind = ReductionKind::Log2;
    else if (isIdentifier("exp2"))
      call.kind = ReductionKind::Exp2;
    else
      call.kind = ReductionKind::Lowpass;
    call.position = current.position;
    advance();
    if (!expect(TokenKind::LeftParen, "expected '(' after builtin"))
      return std::nullopt;
    const ParameterAst *firstOperand = findParameter(current.spelling);
    SourceType policyType =
        requestedPolicy.value_or(firstOperand ? firstOperand->type : SourceType::F32);
    if (call.kind == ReductionKind::Lowpass) {
      // The design is fully named by its attributes; the tap count also
      // names the coefficient extent, which no result type spells here.
      if (!expectIdentifier("taps", "expected lowpass tap count") ||
          !expect(TokenKind::Equal, "expected '=' after taps"))
        return std::nullopt;
      auto taps = parseSignedInteger("expected lowpass tap count");
      if (!taps || !expect(TokenKind::Comma, "expected ',' before lowpass cutoff") ||
          !expectIdentifier("cutoff", "expected lowpass cutoff") ||
          !expect(TokenKind::Equal, "expected '=' after cutoff") ||
          !parseRationalConstant("cutoff", call.cutoffNum, call.cutoffDen) ||
          !expect(TokenKind::RightParen, "expected ')' after lowpass expression"))
        return std::nullopt;
      call.taps = *taps;
      return call;
    }
    if (isWindowDesignKind(call.kind)) {
      // Same shape as lowpass: the design is fully named by its attributes and
      // the tap count also names the coefficient extent.
      if (!expectIdentifier("taps", "expected window tap count") ||
          !expect(TokenKind::Equal, "expected '=' after taps"))
        return std::nullopt;
      auto taps = parseSignedInteger("expected window tap count");
      if (!taps)
        return std::nullopt;
      call.taps = *taps;
      if (call.kind == ReductionKind::Kaiser) {
        if (!expect(TokenKind::Comma, "expected ',' before kaiser beta") ||
            !expectIdentifier("beta", "expected kaiser beta") ||
            !expect(TokenKind::Equal, "expected '=' after beta") ||
            !parseRationalConstant("beta", call.betaNum, call.betaDen))
          return std::nullopt;
      }
      if (!expect(TokenKind::RightParen, "expected ')' after window design expression"))
        return std::nullopt;
      return call;
    }
    if (isFftKind(call.kind)) {
      std::optional<ExpressionAst> operand = parseExpression(std::nullopt);
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      // The interleaved floating-point profile has no stage boundary, so its
      // one axis is the contract mode; sema decides which parameter set the
      // operand element type admits.
      if (current.kind == TokenKind::Comma && next.spelling == "contract") {
        if (!expect(TokenKind::Comma, "expected ',' before floating-point contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
      }
      // Only the CFFT profile admits a declared stage policy; one pair names
      // both scale boundaries, the shape the gated combinations share.
      // Omission keeps the nearest_even saturating default.
      if (isCfftKind(call.kind) && current.kind == TokenKind::Comma &&
          next.spelling == "rounding") {
        if (!expect(TokenKind::Comma, "expected ',' before rounding policy") ||
            !expectIdentifier("rounding", "expected rounding policy") ||
            !expect(TokenKind::Equal, "expected '=' after rounding"))
          return std::nullopt;
        auto rounding = parseIdentifier("expected rounding mode");
        if (!rounding)
          return std::nullopt;
        call.rounding = rounding->spelling.str();
      }
      if (isCfftKind(call.kind) && current.kind == TokenKind::Comma) {
        if (!expect(TokenKind::Comma, "expected ',' before overflow policy") ||
            !expectIdentifier("overflow", "expected overflow policy") ||
            !expect(TokenKind::Equal, "expected '=' after overflow"))
          return std::nullopt;
        auto overflow = parseIdentifier("expected overflow mode");
        if (!overflow)
          return std::nullopt;
        call.destinationOverflow = overflow->spelling.str();
      }
      if (!expect(TokenKind::RightParen, "expected ')' after FFT operand"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::ViterbiDecode) {
      std::optional<ExpressionAst> operand = parseExpression(std::nullopt);
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      if (!parseNamedInteger("constraint_length", call.constraintLength) ||
          !expect(TokenKind::Comma, "expected ',' before polynomials") ||
          !expectIdentifier("polynomials", "expected polynomials") ||
          !expect(TokenKind::Equal, "expected '=' after polynomials") ||
          !expect(TokenKind::LeftBracket, "expected '[' before the generator polynomials"))
        return std::nullopt;
      while (true) {
        std::optional<int64_t> polynomial = parseSignedInteger("expected a generator polynomial");
        if (!polynomial)
          return std::nullopt;
        call.polynomials.push_back(*polynomial);
        if (current.kind != TokenKind::Comma)
          break;
        advance();
      }
      if (!expect(TokenKind::RightBracket, "expected ']' after the generator polynomials"))
        return std::nullopt;
      // A declared precondition on the symbols, which the decoding does not
      // depend on and a narrower metric certificate may read.
      if (current.kind == TokenKind::Comma) {
        advance();
        if (!expectIdentifier("symbol_bound", "expected symbol_bound") ||
            !expect(TokenKind::Equal, "expected '=' after symbol_bound"))
          return std::nullopt;
        SourcePosition position = current.position;
        auto bound = parseSignedInteger("expected a symbol_bound in raw symbol units");
        if (!bound)
          return std::nullopt;
        if (*bound < 1 || *bound > 32767) {
          diagnostics.error(position, "symbol_bound must lie in [1, 32767]");
          return std::nullopt;
        }
        call.symbolBound = *bound;
      }
      if (!expect(TokenKind::RightParen, "expected ')' after viterbi_decode expression"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::Phase) {
      // The phase contract admits exactly one tie rule, so the only choice at
      // the call site is the turn width; omission keeps the Q0.16 turn.
      std::optional<ExpressionAst> operand = parseExpression(std::nullopt);
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      if (current.kind == TokenKind::Comma) {
        advance();
        // The interleaved f32 turn is the format's own, so that profile spells
        // a contract mode where the fixed widths spell a turn width.
        if (isIdentifier("contract")) {
          advance();
          if (!expect(TokenKind::Equal, "expected '=' after contract"))
            return std::nullopt;
          std::optional<Token> contract = parseIdentifier("expected floating-point contract mode");
          if (!contract)
            return std::nullopt;
          call.fpContract = contract->spelling.str();
        } else {
          std::optional<Token> name = parseIdentifier("expected phase turn width");
          if (!name || name->spelling != "turn" ||
              !expect(TokenKind::Equal, "expected '=' after turn")) {
            if (name && name->spelling != "turn")
              diagnostics.error(name->position,
                                "phase accepts only turn=q15, turn=q31, or contract=");
            return std::nullopt;
          }
          std::optional<Token> width = parseIdentifier("expected phase turn width");
          if (!width)
            return std::nullopt;
          if (width->spelling != "q15" && width->spelling != "q31") {
            diagnostics.error(width->position, "phase accepts only turn=q15 or turn=q31");
            return std::nullopt;
          }
          call.turn = width->spelling.str();
        }
      }
      if (!expect(TokenKind::RightParen, "expected ')' after phase operand"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::Magnitude) {
      std::optional<ExpressionAst> operand = parseExpression(std::nullopt);
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      // Two boundaries, two parameters, each optional and each defaulting to
      // nearest_even: the component pre-shift the wider widths carry, and the
      // root. Sema decides which of them the operand's width actually has.
      while (current.kind == TokenKind::Comma) {
        advance();
        // The interleaved f32 profile has neither boundary, so its one axis
        // is the contract mode; sema decides which set the operand admits.
        if (isIdentifier("contract")) {
          advance();
          if (!expect(TokenKind::Equal, "expected '=' after contract"))
            return std::nullopt;
          auto contract = parseIdentifier("expected floating-point contract mode");
          if (!contract)
            return std::nullopt;
          call.fpContract = contract->spelling.str();
          continue;
        }
        bool isRoot = isIdentifier("root_rounding");
        if (!isRoot && !isIdentifier("input_rounding")) {
          diagnostics.error(current.position,
                            "expected root_rounding, input_rounding, or contract policy");
          return std::nullopt;
        }
        advance();
        if (!expect(TokenKind::Equal, "expected '=' after the magnitude rounding policy"))
          return std::nullopt;
        auto rounding = parseIdentifier("expected rounding mode");
        if (!rounding)
          return std::nullopt;
        (isRoot ? call.rounding : call.inputRounding) = rounding->spelling.str();
      }
      if (!expect(TokenKind::RightParen, "expected ')' after magnitude operand"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::Power) {
      std::optional<ExpressionAst> operand = parseExpression(std::nullopt);
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      // One choice either way: the fixed sum is exact and only its reading is
      // open, and the f32 sum has no reading but does have a contract mode.
      if (current.kind == TokenKind::Comma) {
        advance();
        bool isContract = isIdentifier("contract");
        if (!isContract && !isIdentifier("rounding")) {
          diagnostics.error(current.position, "expected rounding or contract policy");
          return std::nullopt;
        }
        advance();
        if (!expect(TokenKind::Equal, "expected '=' after the power policy"))
          return std::nullopt;
        auto mode = parseIdentifier(isContract ? "expected floating-point contract mode"
                                               : "expected rounding mode");
        if (!mode)
          return std::nullopt;
        (isContract ? call.fpContract : call.rounding) = mode->spelling.str();
      }
      if (!expect(TokenKind::RightParen, "expected ')' after power operand"))
        return std::nullopt;
      return call;
    }
    if (isConversionKind(call.kind)) {
      // The target is spelled although two widths leave one choice: the
      // conversion is the only place a program changes width.
      std::optional<ExpressionAst> input = parseExpression(std::nullopt);
      if (!input)
        return std::nullopt;
      call.operands.push_back(std::move(*input));
      if (!expect(TokenKind::Comma, "expected ',' before the conversion target") ||
          !expectIdentifier("to", "expected 'to' naming the conversion target") ||
          !expect(TokenKind::Equal, "expected '=' after to"))
        return std::nullopt;
      std::optional<SourceType> target =
          parseSourceType("expected the target format 'q15', 'q31', or 'f32'");
      if (!target)
        return std::nullopt;
      call.target = *target;
      if (!isLossyConversionKind(call.kind) && current.kind == TokenKind::Comma) {
        diagnostics.error(current.position,
                          call.kind == ReductionKind::Widen
                              ? "widen is exact and takes no rounding or overflow policy"
                              : "dequantize is one IEEE rounding and takes no rounding or "
                                "overflow policy");
        return std::nullopt;
      }
      if (!parseBoundaryPolicies(call) ||
          !expect(TokenKind::RightParen, "expected ')' after the conversion"))
        return std::nullopt;
      return call;
    }
    if (isElementwiseKind(call.kind)) {
      std::optional<ExpressionAst> lhs = parseExpression(std::nullopt);
      if (!lhs)
        return std::nullopt;
      call.operands.push_back(std::move(*lhs));
      if (isBinaryElementwiseKind(call.kind)) {
        if (!expect(TokenKind::Comma, "expected ',' before the second elementwise operand"))
          return std::nullopt;
        std::optional<ExpressionAst> rhs = parseExpression(std::nullopt);
        if (!rhs)
          return std::nullopt;
        call.operands.push_back(std::move(*rhs));
      }
      if (call.kind == ReductionKind::Offset && !parseNamedInteger("bias", call.bias))
        return std::nullopt;
      if (call.kind == ReductionKind::Shift && !parseNamedInteger("amount", call.amount))
        return std::nullopt;
      if (call.kind == ReductionKind::Div && !parseNamedInteger("divisor", call.divisor))
        return std::nullopt;
      if (!parseBoundaryPolicies(call))
        return std::nullopt;
      if (call.kind == ReductionKind::Ratio && current.kind == TokenKind::Comma) {
        if (!expect(TokenKind::Comma, "expected ',' before the nonpositive divisor policy") ||
            !expectIdentifier("nonpositive", "expected the nonpositive divisor policy") ||
            !expect(TokenKind::Equal, "expected '=' after nonpositive"))
          return std::nullopt;
        auto policy = parseIdentifier("expected trap or saturate");
        if (!policy)
          return std::nullopt;
        call.nonpositive = policy->spelling.str();
      }
      if (!expect(TokenKind::RightParen, "expected ')' after elementwise expression"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::Gain) {
      // The fixed gain nests, so its operand is any composable expression.
      std::optional<ExpressionAst> input = parseExpression(std::nullopt);
      if (!input)
        return std::nullopt;
      call.operands.push_back(std::move(*input));
    } else {
      auto lhs = parseIdentifier("expected builtin operand");
      if (!lhs)
        return std::nullopt;
      call.operands.push_back(resolveOperand(*lhs));
    }
    if (call.kind == ReductionKind::SosDf2Fixed) {
      if (!expect(TokenKind::Comma, "expected ',' after sos_df2_fixed input operand"))
        return std::nullopt;
      auto coefficients = parseIdentifier("expected sos_df2_fixed coefficients operand");
      if (!coefficients ||
          !expect(TokenKind::Comma, "expected ',' after sos_df2_fixed coefficients operand"))
        return std::nullopt;
      auto scales = parseIdentifier("expected sos_df2_fixed scales operand");
      if (!scales || !expect(TokenKind::Comma, "expected ',' after sos_df2_fixed scales operand"))
        return std::nullopt;
      auto state = parseIdentifier("expected sos_df2_fixed state operand");
      if (!state)
        return std::nullopt;
      call.operands.push_back(resolveOperand(*coefficients));
      call.operands.push_back(resolveOperand(*scales));
      call.operands.push_back(resolveOperand(*state));
      if (current.kind == TokenKind::RightParen) {
        applyDefaultSosPolicy(call);
      } else if (!expect(TokenKind::Comma, "expected ',' before sos_df2_fixed numeric policy") ||
                 !parseFixedSosPolicy(call)) {
        return std::nullopt;
      }
      if (!expect(TokenKind::RightParen, "expected ')' after sos_df2_fixed expression"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::SosTdf2) {
      if (!expect(TokenKind::Comma, "expected ',' after sos_tdf2 input operand"))
        return std::nullopt;
      auto coefficients = parseIdentifier("expected sos_tdf2 coefficients operand");
      if (!coefficients || !expect(TokenKind::Comma, "expected ',' after sos_tdf2 coefficients "
                                                     "operand"))
        return std::nullopt;
      auto scales = parseIdentifier("expected sos_tdf2 scales operand");
      if (!scales || !expect(TokenKind::Comma, "expected ',' after sos_tdf2 scales operand"))
        return std::nullopt;
      auto state = parseIdentifier("expected sos_tdf2 state operand");
      if (!state || !expect(TokenKind::Comma, "expected ',' before sos_tdf2 contract policy") ||
          !expectIdentifier("contract", "expected floating-point contract policy") ||
          !expect(TokenKind::Equal, "expected '=' after contract"))
        return std::nullopt;
      auto contract = parseIdentifier("expected floating-point contract mode");
      if (!contract || !expect(TokenKind::RightParen, "expected ')' after sos_tdf2 expression"))
        return std::nullopt;
      call.operands.push_back(resolveOperand(*coefficients));
      call.operands.push_back(resolveOperand(*scales));
      call.operands.push_back(resolveOperand(*state));
      call.fpContract = contract->spelling.str();
      return call;
    }
    if (call.kind == ReductionKind::Goertzel) {
      if (!parseNamedInteger("bin", call.bin) ||
          !expect(TokenKind::Comma, "expected ',' before goertzel contract policy") ||
          !expectIdentifier("contract", "expected floating-point contract policy") ||
          !expect(TokenKind::Equal, "expected '=' after contract"))
        return std::nullopt;
      auto contract = parseIdentifier("expected floating-point contract mode");
      if (!contract || !expect(TokenKind::RightParen, "expected ')' after goertzel expression"))
        return std::nullopt;
      call.fpContract = contract->spelling.str();
      return call;
    }
    if (call.kind == ReductionKind::Lms) {
      if (!expect(TokenKind::Comma, "expected ',' after lms input operand"))
        return std::nullopt;
      auto desired = parseIdentifier("expected lms desired operand");
      if (!desired || !expect(TokenKind::Comma, "expected ',' after lms desired operand"))
        return std::nullopt;
      auto weights = parseIdentifier("expected lms weights operand");
      if (!weights || !expect(TokenKind::Comma, "expected ',' before lms step size") ||
          !expectIdentifier("step_size", "expected lms step size") ||
          !expect(TokenKind::Equal, "expected '=' after step_size"))
        return std::nullopt;
      if (policyType == SourceType::F32) {
        if (!parseRationalConstant("step size", call.stepSize, call.fpConstantDen) ||
            !expect(TokenKind::Comma, "expected ',' before lms contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
      } else {
        auto stepSize = parseSignedInteger("expected lms step size");
        if (!stepSize)
          return std::nullopt;
        call.stepSize = *stepSize;
        // The normalized recursion divides the step by the window energy plus
        // epsilon; the constant is the profile's, so lms refuses it and nlms
        // requires it.
        if (call.normalized) {
          if (!parseNamedInteger("epsilon", call.epsilon))
            return std::nullopt;
        } else if (current.kind == TokenKind::Comma && next.spelling == "epsilon") {
          diagnostics.error(next.position,
                            "epsilon is the normalized profile's constant; spell nlms(...)");
          return std::nullopt;
        }
        // The Q31 tap sum carries a per-product boundary; its rounding is
        // declared here, beside the step size, and refused where the sum
        // already fits. Bindings must expose every choice their contract
        // admits.
        if (current.kind == TokenKind::Comma) {
          if (!expect(TokenKind::Comma, "expected ',' before lms product rounding policy") ||
              !expectIdentifier("product_rounding", "expected lms product rounding policy") ||
              !expect(TokenKind::Equal, "expected '=' after product_rounding"))
            return std::nullopt;
          auto rounding = parseIdentifier("expected rounding mode");
          if (!rounding)
            return std::nullopt;
          call.inputRounding = rounding->spelling.str();
        }
      }
      if (!expect(TokenKind::RightParen, "expected ')' after lms expression"))
        return std::nullopt;
      call.operands.push_back(resolveOperand(*desired));
      call.operands.push_back(resolveOperand(*weights));
      return call;
    }
    if (call.kind == ReductionKind::Butterfly || call.kind == ReductionKind::FirStream) {
      llvm::StringRef builtin = call.kind == ReductionKind::Butterfly ? "butterfly" : "fir_stream";
      if (!expect(TokenKind::Comma,
                  llvm::Twine("expected ',' after ") + builtin + " first operand"))
        return std::nullopt;
      auto rhs = parseIdentifier(llvm::Twine("expected ") + builtin + " second operand");
      if (!rhs || !expect(TokenKind::Comma,
                          llvm::Twine("expected ',' after ") + builtin + " second operand"))
        return std::nullopt;
      auto third = parseIdentifier(call.kind == ReductionKind::Butterfly
                                       ? "expected butterfly twiddle operand"
                                       : "expected fir_stream state operand");
      if (!third)
        return std::nullopt;
      call.operands.push_back(resolveOperand(*rhs));
      call.operands.push_back(resolveOperand(*third));
      if (call.kind == ReductionKind::FirStream) {
        if (current.kind == TokenKind::RightParen)
          applyDefaultFixedPolicy(call);
        else if (!expect(TokenKind::Comma, "expected ',' before fir_stream numeric policy") ||
                 !parseFixedPolicy(call))
          return std::nullopt;
      }
      if (!expect(TokenKind::RightParen,
                  llvm::Twine("expected ')' after ") + builtin + " operands"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::Dct || call.kind == ReductionKind::Rms ||
        call.kind == ReductionKind::Sine || call.kind == ReductionKind::Cosine ||
        call.kind == ReductionKind::Log2 || call.kind == ReductionKind::Exp2) {
      llvm::StringRef builtin = call.kind == ReductionKind::Dct    ? "dct"
                                : call.kind == ReductionKind::Rms  ? "rms"
                                : call.kind == ReductionKind::Sine ? "sine"
                                : call.kind == ReductionKind::Log2 ? "log2"
                                : call.kind == ReductionKind::Exp2 ? "exp2"
                                                                   : "cosine";
      if (call.kind == ReductionKind::Dct && policyType == SourceType::F32) {
        if (!expect(TokenKind::Comma, "expected ',' before dct contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
      } else if (call.kind == ReductionKind::Dct && current.kind == TokenKind::Comma) {
        // Each parameter names the boundary it rounds: the export exists at
        // both widths, the per-product boundary only where the row sum would
        // leave i64, which is every Q31 extent. Omission keeps nearest_even.
        while (current.kind == TokenKind::Comma) {
          if (!expect(TokenKind::Comma, "expected ',' before a rounding policy"))
            return std::nullopt;
          std::optional<Token> name = parseIdentifier("expected a rounding policy name");
          if (!name)
            return std::nullopt;
          bool isExport = name->spelling == "rounding";
          bool isProduct = name->spelling == "product";
          if (!isExport && !isProduct && name->spelling != "product_rounding") {
            diagnostics.error(name->position, "dct accepts product, rounding and product_rounding");
            return std::nullopt;
          }
          if (!expect(TokenKind::Equal, "expected '=' after a dct policy name"))
            return std::nullopt;
          auto value = parseIdentifier(isProduct ? "expected product selection 'full' or 'raw_high'"
                                                 : "expected rounding mode");
          if (!value)
            return std::nullopt;
          (isProduct  ? call.product
           : isExport ? call.rounding
                      : call.inputRounding) = value->spelling.str();
        }
      }
      if (call.kind == ReductionKind::Rms && policyType == SourceType::F32) {
        if (!expect(TokenKind::Comma, "expected ',' before rms contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
      } else if (call.kind == ReductionKind::Rms && current.kind == TokenKind::Comma) {
        // Each parameter names the specific boundary it rounds, because rms
        // carries two of them at Q15 and three at Q31, where the input is
        // requantized before squaring. The mean boundary stays nearest even.
        // Omission keeps the default. Bindings must expose every choice their
        // contract admits.
        while (current.kind == TokenKind::Comma) {
          if (!expect(TokenKind::Comma, "expected ',' before a rounding policy"))
            return std::nullopt;
          std::optional<Token> name = parseIdentifier("expected a rounding policy name");
          if (!name)
            return std::nullopt;
          bool isRoot = name->spelling == "root_rounding";
          if (!isRoot && name->spelling != "input_rounding") {
            diagnostics.error(name->position, "rms accepts root_rounding and input_rounding");
            return std::nullopt;
          }
          if (!expect(TokenKind::Equal, "expected '=' after a rounding policy name"))
            return std::nullopt;
          auto rounding = parseIdentifier("expected rounding mode");
          if (!rounding)
            return std::nullopt;
          (isRoot ? call.rounding : call.inputRounding) = rounding->spelling.str();
        }
      }
      if (!expect(TokenKind::RightParen, llvm::Twine("expected ')' after ") + builtin + " operand"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::CicDecimate) {
      // state_overflow has no default: the cascade is only correct under
      // wrap, so the declaration is the source's, never the compiler's.
      if (!parseNamedInteger("stages", call.stages) || !parseNamedInteger("rate", call.rate) ||
          !parseNamedInteger("delay", call.delay))
        return std::nullopt;
      if (!expect(TokenKind::Comma, "expected ',' before state_overflow policy") ||
          !expectIdentifier("state_overflow", "expected state_overflow policy") ||
          !expect(TokenKind::Equal, "expected '=' after state_overflow"))
        return std::nullopt;
      auto overflow = parseIdentifier("expected state overflow mode");
      if (!overflow)
        return std::nullopt;
      call.stateOverflow = overflow->spelling.str();
      if (current.kind == TokenKind::Comma) {
        if (!expect(TokenKind::Comma, "expected ',' before rounding policy") ||
            !expectIdentifier("rounding", "expected rounding policy") ||
            !expect(TokenKind::Equal, "expected '=' after rounding"))
          return std::nullopt;
        auto rounding = parseIdentifier("expected rounding mode");
        if (!rounding)
          return std::nullopt;
        call.rounding = rounding->spelling.str();
      }
      if (!expect(TokenKind::RightParen, "expected ')' after cic_decimate expression"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::MovingAverage || call.kind == ReductionKind::Gain) {
      bool isGain = call.kind == ReductionKind::Gain;
      llvm::StringRef builtin = isGain ? "gain" : "moving_average";
      llvm::StringRef keyword = isGain ? "gain" : "window";
      if (!expect(TokenKind::Comma, llvm::Twine("expected ',' before ") + builtin + " constant") ||
          !expectIdentifier(keyword, llvm::Twine("expected ") + builtin + " constant") ||
          !expect(TokenKind::Equal, llvm::Twine("expected '=' after ") + keyword))
        return std::nullopt;
      if (isGain && policyType == SourceType::F32) {
        if (!parseRationalConstant("gain", call.gain, call.fpConstantDen) ||
            !expect(TokenKind::Comma, "expected ',' before gain contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
        if (!expect(TokenKind::RightParen, "expected ')' after gain expression"))
          return std::nullopt;
        return call;
      }
      auto constant = parseSignedInteger(llvm::Twine("expected ") + builtin + " constant");
      if (!constant)
        return std::nullopt;
      if (isGain)
        call.gain = *constant;
      else
        call.window = *constant;
      if (!isGain && policyType == SourceType::F32) {
        if (!expect(TokenKind::Comma, "expected ',' before moving_average contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
      }
      if (isGain && current.kind == TokenKind::Comma) {
        // gain has a single requantization boundary, so the parameter is the
        // plain `rounding=`, not boundary-qualified; omission takes the
        // export default (nearest_ties_positive).
        if (!expect(TokenKind::Comma, "expected ',' before rounding policy") ||
            !expectIdentifier("rounding", "expected rounding policy") ||
            !expect(TokenKind::Equal, "expected '=' after rounding"))
          return std::nullopt;
        auto rounding = parseIdentifier("expected rounding mode");
        if (!rounding)
          return std::nullopt;
        call.rounding = rounding->spelling.str();
      }
      if (!expect(TokenKind::RightParen,
                  llvm::Twine("expected ')' after ") + builtin + " expression"))
        return std::nullopt;
      return call;
    }
    if (!expect(TokenKind::Comma, "expected ',' after reduction left operand"))
      return std::nullopt;
    auto rhs = parseIdentifier("expected reduction right operand");
    if (!rhs)
      return std::nullopt;
    call.operands.push_back(resolveOperand(*rhs));

    // Which operand is conjugated is the difference between a complex dot
    // product and a correlation, so the call site names it; omission is the
    // plain product.
    if ((call.kind == ReductionKind::CxDot || call.kind == ReductionKind::CxFirFilter) &&
        current.kind == TokenKind::Comma && next.spelling == "conjugate") {
      if (!expect(TokenKind::Comma, "expected ',' before conjugate selection") ||
          !expectIdentifier("conjugate", "expected conjugate selection") ||
          !expect(TokenKind::Equal, "expected '=' after conjugate"))
        return std::nullopt;
      auto conjugate = parseIdentifier("expected conjugate selection 'true' or 'false'");
      if (!conjugate)
        return std::nullopt;
      if (conjugate->spelling != "true" && conjugate->spelling != "false") {
        diagnostics.error(conjugate->position,
                          "a complex reduction accepts only conjugate=true or conjugate=false");
        return std::nullopt;
      }
      call.conjugate = conjugate->spelling == "true";
    }

    if (call.kind == ReductionKind::Matmul) {
      if (policyType == SourceType::F32) {
        if (!expect(TokenKind::Comma, "expected ',' before matmul contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
      } else if (current.kind == TokenKind::Comma) {
        // The export boundary always admits a declared rounding; the Q31
        // profile adds a product boundary whose rounding is declared beside
        // it. Omission keeps the nearest_even default. Bindings must expose
        // every choice their contract admits.
        while (current.kind == TokenKind::Comma) {
          if (!expect(TokenKind::Comma, "expected ',' before a matmul rounding policy"))
            return std::nullopt;
          std::optional<Token> name = parseIdentifier("expected a matmul rounding policy name");
          if (!name)
            return std::nullopt;
          bool isExport = name->spelling == "rounding";
          bool isProduct = name->spelling == "product";
          if (!isExport && !isProduct && name->spelling != "product_rounding") {
            diagnostics.error(name->position,
                              "matmul accepts product, rounding and product_rounding");
            return std::nullopt;
          }
          if (!expect(TokenKind::Equal, "expected '=' after a matmul policy name"))
            return std::nullopt;
          auto value = parseIdentifier(isProduct ? "expected product selection 'full' or 'raw_high'"
                                                 : "expected rounding mode");
          if (!value)
            return std::nullopt;
          (isProduct  ? call.product
           : isExport ? call.rounding
                      : call.inputRounding) = value->spelling.str();
        }
      }
      if (!expect(TokenKind::RightParen, "expected ')' after matmul expression"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::FirFilter || call.kind == ReductionKind::CxFirFilter) {
      if (!expect(TokenKind::Comma, "expected ',' before FIR boundary policy"))
        return std::nullopt;
      if (!expectIdentifier("boundary", "expected FIR boundary policy") ||
          !expect(TokenKind::Equal, "expected '=' after boundary"))
        return std::nullopt;
      auto boundary = parseIdentifier("expected FIR boundary mode");
      if (!boundary)
        return std::nullopt;
      call.boundary = boundary->spelling.str();
      // A fixed call site that stops at the boundary takes the same default
      // contract the plain reductions take; f32 falls through to its
      // mandatory contract, which has no defensible default.
      if (current.kind == TokenKind::RightParen) {
        if (policyType != SourceType::F32)
          applyDefaultFixedPolicy(call);
      } else if (!expect(TokenKind::Comma, "expected ',' after FIR boundary mode")) {
        return std::nullopt;
      }
    } else if (call.kind == ReductionKind::FirDecimate ||
               call.kind == ReductionKind::FirInterpolate) {
      bool isInterpolation = call.kind == ReductionKind::FirInterpolate;
      llvm::StringRef operation = isInterpolation ? "fir_interpolate" : "fir_decimate";
      if (!expect(TokenKind::Comma, "expected ',' before FIR resampling factor") ||
          !expectIdentifier("factor", "expected FIR resampling factor") ||
          !expect(TokenKind::Equal, "expected '=' after factor"))
        return std::nullopt;
      auto factor = parseSignedInteger("expected FIR resampling factor");
      if (!factor)
        return std::nullopt;
      call.factor = *factor;
      if (policyType == SourceType::F32) {
        if (!expect(TokenKind::Comma,
                    llvm::Twine("expected ',' before ") + operation + " contract policy") ||
            !expectIdentifier("contract", "expected floating-point contract policy") ||
            !expect(TokenKind::Equal, "expected '=' after contract"))
          return std::nullopt;
        auto contract = parseIdentifier("expected floating-point contract mode");
        if (!contract)
          return std::nullopt;
        call.fpContract = contract->spelling.str();
      } else if (current.kind == TokenKind::Comma) {
        // The explicit triple, which is what a Q31 resampling site needs:
        // automatic accumulation is exact and no exact accumulator exists for
        // K products of 62 bits, so that width must declare its own carrier
        // and per-update overflow mode.
        if (!expect(TokenKind::Comma,
                    llvm::Twine("expected ',' before ") + operation + " numeric policy") ||
            !parseFixedPolicy(call))
          return std::nullopt;
      } else {
        applyDefaultFixedPolicy(call);
      }
      if (!expect(TokenKind::RightParen,
                  llvm::Twine("expected ')' after ") + operation + " expression"))
        return std::nullopt;
      return call;
    } else if (current.kind == TokenKind::RightParen && !isFpPolicyType(policyType)) {
      applyDefaultFixedPolicy(call);
    } else if (!expect(TokenKind::Comma, "expected ',' before numeric policy")) {
      return std::nullopt;
    }
    // A declared precondition on the coefficients, spelled before the numeric
    // policy it does not change; left alone, the policy keeps its default.
    if (call.kind == ReductionKind::Dot && isIdentifier("gain_bound")) {
      if (isFpPolicyType(policyType)) {
        diagnostics.error(current.position,
                          "gain_bound is a fixed-point coefficient precondition; an f32 dot "
                          "has no accumulator range for it to bound");
        return std::nullopt;
      }
      advance();
      if (!expect(TokenKind::Equal, "expected '=' after gain_bound"))
        return std::nullopt;
      SourcePosition position = current.position;
      auto bound = parseSignedInteger("expected a gain_bound in full-scale units");
      if (!bound)
        return std::nullopt;
      if (*bound < 1 || *bound > 65536) {
        diagnostics.error(position, "gain_bound must lie in [1, 65536]");
        return std::nullopt;
      }
      call.gainBound = *bound;
      if (current.kind == TokenKind::RightParen)
        applyDefaultFixedPolicy(call);
      else if (!expect(TokenKind::Comma, "expected ',' before numeric policy"))
        return std::nullopt;
    }

    if (!isFpPolicyType(policyType)) {
      if (!call.accumulatorAuto && !parseFixedPolicy(call))
        return std::nullopt;
    } else {
      if (!expectIdentifier("contract", "expected floating-point contract policy") ||
          !expect(TokenKind::Equal, "expected '=' after contract"))
        return std::nullopt;
      auto contract = parseIdentifier("expected floating-point contract mode");
      if (!contract)
        return std::nullopt;
      call.fpContract = contract->spelling.str();
    }

    if (!expect(TokenKind::RightParen, "expected ')' after reduction expression"))
      return std::nullopt;

    return call;
  }

private:
  // A statement is a builtin call or an infix expression over them; a bare
  // parameter name would be an alias, which the language does not have.
  std::optional<BuiltinCallAst> parseStatementExpression(SourceType policyType) {
    std::optional<Operand> operand =
        genericExpressions ? parseLogicalOr(policyType) : parseSum(policyType);
    if (!operand)
      return std::nullopt;
    if (genericExpressions && operand->expression && operand->expression->isParameterReference())
      return makePrimitive(PrimitiveKind::Identity, {&*operand->expression}, operand->position);
    if (!operand->expression || operand->expression->isParameterReference()) {
      diagnostics.error(operand->position, "expected a builtin call or an infix expression, not "
                                           "a bare name or literal");
      return std::nullopt;
    }
    return std::move(*operand->expression->call);
  }

  // An operand of a composable builtin: a whole infix expression, never a
  // literal on its own, which only an operator with a tensor beside it reads.
  std::optional<ExpressionAst> parseExpression(std::optional<SourceType> rootPolicy) {
    std::optional<Operand> operand =
        genericExpressions ? parseLogicalOr(rootPolicy) : parseSum(rootPolicy);
    if (!operand)
      return std::nullopt;
    if (!operand->expression) {
      diagnostics.error(operand->position, "a literal needs a tensor operand beside it");
      return std::nullopt;
    }
    return std::move(operand->expression);
  }

  static BuiltinCallAst makePrimitive(PrimitiveKind kind,
                                      std::initializer_list<ExpressionAst *> operands,
                                      SourcePosition position) {
    BuiltinCallAst call;
    call.kind = ReductionKind::Primitive;
    call.primitive = kind;
    call.position = position;
    for (ExpressionAst *operand : operands)
      call.operands.push_back(std::move(*operand));
    return call;
  }

  std::optional<Operand> parseLogicalOr(std::optional<SourceType> rootPolicy = std::nullopt) {
    auto lhs = parseLogicalAnd(rootPolicy);
    while (lhs && isIdentifier("or")) {
      SourcePosition position = current.position;
      advance();
      auto rhs = parseLogicalAnd();
      if (!rhs)
        return std::nullopt;
      lhs = Operand{ExpressionAst(makePrimitive(PrimitiveKind::Or,
                                                {&*lhs->expression, &*rhs->expression}, position)),
                    0, position};
    }
    return lhs;
  }
  std::optional<Operand> parseLogicalAnd(std::optional<SourceType> rootPolicy = std::nullopt) {
    auto lhs = parseLogicalNot(rootPolicy);
    while (lhs && isIdentifier("and")) {
      SourcePosition position = current.position;
      advance();
      auto rhs = parseLogicalNot();
      if (!rhs)
        return std::nullopt;
      lhs = Operand{ExpressionAst(makePrimitive(PrimitiveKind::And,
                                                {&*lhs->expression, &*rhs->expression}, position)),
                    0, position};
    }
    return lhs;
  }
  std::optional<Operand> parseLogicalNot(std::optional<SourceType> rootPolicy = std::nullopt) {
    if (isIdentifier("not")) {
      SourcePosition position = current.position;
      advance();
      auto value = parseLogicalNot();
      if (!value)
        return std::nullopt;
      return Operand{
          ExpressionAst(makePrimitive(PrimitiveKind::Not, {&*value->expression}, position)), 0,
          position};
    }
    auto lhs = parseSum(rootPolicy);
    PrimitiveKind op;
    switch (current.kind) {
    case TokenKind::EqualEqual:
      op = PrimitiveKind::Eq;
      break;
    case TokenKind::NotEqual:
      op = PrimitiveKind::Ne;
      break;
    case TokenKind::Less:
      op = PrimitiveKind::Lt;
      break;
    case TokenKind::LessEqual:
      op = PrimitiveKind::Le;
      break;
    case TokenKind::Greater:
      op = PrimitiveKind::Gt;
      break;
    case TokenKind::GreaterEqual:
      op = PrimitiveKind::Ge;
      break;
    default:
      return lhs;
    }
    SourcePosition position = current.position;
    advance();
    auto rhs = parseSum(std::nullopt);
    if (!lhs || !rhs)
      return std::nullopt;
    return Operand{
        ExpressionAst(makePrimitive(op, {&*lhs->expression, &*rhs->expression}, position)), 0,
        position};
  }

  // Infix `+`, `-`, `*` and `/` are `add`, `sub`, `mult` and `div` under the
  // language defaults: `*` and `/` bind tighter, all associate to the left,
  // parentheses group, and every operator is its own quantization boundary.
  std::optional<Operand> parseSum(std::optional<SourceType> rootPolicy) {
    std::optional<Operand> lhs = parseTerm(rootPolicy);
    while (lhs && (current.kind == TokenKind::Plus || current.kind == TokenKind::Minus)) {
      ReductionKind kind =
          current.kind == TokenKind::Plus ? ReductionKind::Add : ReductionKind::Sub;
      SourcePosition position = current.position;
      advance();
      std::optional<Operand> rhs = parseTerm(std::nullopt);
      if (!rhs)
        return std::nullopt;
      lhs = combine(kind, std::move(*lhs), std::move(*rhs), position);
    }
    return lhs;
  }

  std::optional<Operand> parseTerm(std::optional<SourceType> rootPolicy) {
    auto lhs = parsePrimary(rootPolicy);
    while (lhs && (current.kind == TokenKind::Star || current.kind == TokenKind::Slash)) {
      ReductionKind kind =
          current.kind == TokenKind::Star ? ReductionKind::Mult : ReductionKind::Ratio;
      SourcePosition position = current.position;
      advance();
      auto rhs = parsePrimary(std::nullopt);
      if (!rhs)
        return std::nullopt;
      lhs = combine(kind, std::move(*lhs), std::move(*rhs), position);
    }
    return lhs;
  }

  std::optional<Operand> combine(ReductionKind kind, Operand lhs, Operand rhs,
                                 SourcePosition position) {
    if (genericExpressions) {
      return Operand{ExpressionAst(makePrimitive(*getOperatorPrimitive(kind),
                                                 {&*lhs.expression, &*rhs.expression}, position)),
                     0, position};
    }
    auto call = buildFixedBinary(kind, std::move(lhs), std::move(rhs), position, diagnostics);
    if (!call)
      return std::nullopt;
    return Operand{ExpressionAst(std::move(*call)), 0, position};
  }

  // A primary is a parenthesized expression, a callee instantiation, a builtin
  // call, an operand name or an integer literal. A statement (`rootPolicy`
  // set) may start with any builtin; inside an expression only the composable
  // family nests.
  std::optional<Operand> parsePrimary(std::optional<SourceType> rootPolicy) {
    SourcePosition position = current.position;
    if (current.kind == TokenKind::LeftParen) {
      advance();
      std::optional<Operand> inner = genericExpressions ? parseLogicalOr() : parseSum(std::nullopt);
      if (!inner ||
          !expect(TokenKind::RightParen, "expected ')' to close the parenthesized expression"))
        return std::nullopt;
      return inner;
    }
    if (genericExpressions &&
        (current.kind == TokenKind::Integer || current.kind == TokenKind::Float ||
         isIdentifier("True") || isIdentifier("False") ||
         (current.kind == TokenKind::Minus &&
          (next.kind == TokenKind::Integer || next.kind == TokenKind::Float)))) {
      BuiltinCallAst literal;
      literal.kind = ReductionKind::Primitive;
      literal.primitive = PrimitiveKind::Literal;
      literal.position = position;
      if (current.kind == TokenKind::Minus) {
        literal.spelling = "-";
        advance();
      }
      literal.spelling += current.spelling.str();
      advance();
      return Operand{ExpressionAst(std::move(literal)), 0, position};
    }
    if (current.kind == TokenKind::Plus ||
        (current.kind == TokenKind::Minus && next.kind != TokenKind::Integer)) {
      bool negative = current.kind == TokenKind::Minus;
      advance();
      auto value = parsePrimary(rootPolicy);
      if (!value)
        return std::nullopt;
      if (genericExpressions)
        return Operand{
            ExpressionAst(makePrimitive(negative ? PrimitiveKind::Neg : PrimitiveKind::Pos,
                                        {&*value->expression}, position)),
            0, position};
      if (!value->expression) {
        diagnostics.error(position, "unary operator requires a tensor expression");
        return std::nullopt;
      }
      if (!negative)
        return value;
      BuiltinCallAst call;
      call.kind = ReductionKind::Negate;
      call.position = position;
      call.operands.push_back(std::move(*value->expression));
      return Operand{ExpressionAst(std::move(call)), 0, position};
    }
    if (current.kind == TokenKind::Integer ||
        (current.kind == TokenKind::Minus && next.kind == TokenKind::Integer)) {
      std::optional<int64_t> literal = parseSignedInteger("expected an integer literal");
      if (!literal)
        return std::nullopt;
      return Operand{std::nullopt, *literal, position};
    }
    if (genericExpressions && current.kind == TokenKind::Identifier &&
        next.kind == TokenKind::LeftBracket) {
      Token base = current;
      advance();
      advance();
      std::optional<Operand> index = parseLogicalOr();
      if (!index || !expect(TokenKind::RightBracket, "expected ']' after the index"))
        return std::nullopt;
      ExpressionAst reference = resolveOperand(base);
      return Operand{ExpressionAst(makePrimitive(PrimitiveKind::Element,
                                                 {&reference, &*index->expression}, position)),
                     0, position};
    }
    if (genericExpressions && isIdentifier("len") && next.kind == TokenKind::LeftParen &&
        !calleesByName.contains("len")) {
      advance();
      advance();
      std::optional<Token> base = parseIdentifier("expected a buffer or tensor parameter");
      if (!base || !expect(TokenKind::RightParen, "expected ')' after the len operand"))
        return std::nullopt;
      ExpressionAst reference = resolveOperand(*base);
      return Operand{ExpressionAst(makePrimitive(PrimitiveKind::Length, {&reference}, position)), 0,
                     position};
    }
    bool isCall = current.kind == TokenKind::Identifier && next.kind == TokenKind::LeftParen;
    if (isCall && calleesByName.contains(current.spelling)) {
      // Inside a chain the composition checker establishes element types, so
      // the call is not additionally tied to the function's result type.
      std::optional<BuiltinCallAst> instantiated =
          parseCalleeInstantiation(genericExpressions ? std::nullopt : rootPolicy);
      if (!instantiated)
        return std::nullopt;
      return Operand{ExpressionAst(std::move(*instantiated)), 0, position};
    }
    bool isNestedBuiltin = isIdentifier("cfft") || isIdentifier("icfft") || isIdentifier("rfft") ||
                           isIdentifier("irfft") || isIdentifier("magnitude") ||
                           isIdentifier("add") || isIdentifier("sub") || isIdentifier("mult") ||
                           isIdentifier("abs") || isIdentifier("negate") ||
                           isIdentifier("offset") || isIdentifier("shift") || isIdentifier("div") ||
                           isIdentifier("ratio") || isIdentifier("gain") || isIdentifier("widen") ||
                           isIdentifier("narrow") || isIdentifier("quantize") ||
                           isIdentifier("dequantize") || isIdentifier("viterbi_decode");
    if (isCall && (rootPolicy || isNestedBuiltin || genericExpressions)) {
      std::optional<SourceType> policy = rootPolicy.value_or(SourceType::Q15);
      if (genericExpressions && !isNestedBuiltin)
        policy = std::nullopt;
      std::optional<BuiltinCallAst> call = parseBuiltinCall(policy);
      if (!call)
        return std::nullopt;
      return Operand{ExpressionAst(std::move(*call)), 0, position};
    }
    auto parameter = parseIdentifier(
        "expected an operand name, a nested expression, or a parenthesized expression");
    if (!parameter)
      return std::nullopt;
    return Operand{resolveOperand(*parameter), 0, position};
  }

  bool isIdentifier(llvm::StringRef spelling) const {
    return current.kind == TokenKind::Identifier && current.spelling == spelling;
  }

  void advance() {
    current = next;
    next = lexer.next();
  }

  bool expect(TokenKind kind, const llvm::Twine &message) {
    if (current.kind != kind) {
      diagnostics.error(current.position, message);
      return false;
    }
    advance();
    return true;
  }

  bool expectIdentifier(llvm::StringRef spelling, const llvm::Twine &message) {
    if (!isIdentifier(spelling)) {
      diagnostics.error(current.position, message);
      return false;
    }
    advance();
    return true;
  }

  std::optional<Token> parseIdentifier(const llvm::Twine &message) {
    if (current.kind != TokenKind::Identifier) {
      diagnostics.error(current.position, message);
      return std::nullopt;
    }
    Token result = current;
    advance();
    return result;
  }

  std::optional<Token> parseInteger(const llvm::Twine &message) {
    if (current.kind != TokenKind::Integer) {
      diagnostics.error(current.position, message);
      return std::nullopt;
    }
    Token result = current;
    advance();
    return result;
  }

  std::optional<SourceType> parseSourceType(const llvm::Twine &message) {
    for (auto entry :
         {std::make_pair("i32", SourceType::I32), std::make_pair("i64", SourceType::I64),
          std::make_pair("bool", SourceType::Bool)}) {
      if (isIdentifier(entry.first)) {
        advance();
        return entry.second;
      }
    }
    if (isIdentifier("q15")) {
      advance();
      return SourceType::Q15;
    }
    if (isIdentifier("q31")) {
      advance();
      return SourceType::Q31;
    }
    if (isIdentifier("f32")) {
      advance();
      return SourceType::F32;
    }
    if (isIdentifier("complex_q15")) {
      advance();
      return SourceType::ComplexQ15;
    }
    if (isIdentifier("complex_q31")) {
      advance();
      return SourceType::ComplexQ31;
    }
    if (isIdentifier("complex_f32")) {
      advance();
      return SourceType::ComplexF32;
    }
    if (isIdentifier("u8")) {
      advance();
      return SourceType::U8;
    }
    diagnostics.error(current.position, message);
    return std::nullopt;
  }

  // `[num, den]` is the only spelling for a non-integer constant: the lexer
  // has no floating-point literal, and a rational names the intended value
  // exactly rather than through a decimal the reader must re-round.
  bool parseRationalConstant(const llvm::Twine &name, int64_t &numerator, int64_t &denominator) {
    if (!expect(TokenKind::LeftBracket, "expected '[' before the rational " + name))
      return false;
    auto parsedNumerator = parseSignedInteger("expected " + name + " numerator");
    if (!parsedNumerator ||
        !expect(TokenKind::Comma, "expected ',' between " + name + " numerator and denominator"))
      return false;
    auto parsedDenominator = parseSignedInteger("expected " + name + " denominator");
    if (!parsedDenominator ||
        !expect(TokenKind::RightBracket, "expected ']' after the rational " + name))
      return false;
    numerator = *parsedNumerator;
    denominator = *parsedDenominator;
    return true;
  }

  std::optional<int64_t> parseSignedInteger(const llvm::Twine &message) {
    bool negative = current.kind == TokenKind::Minus;
    if (negative)
      advance();
    auto value = parseInteger(message);
    if (!value)
      return std::nullopt;

    uint64_t magnitude = 0;
    if (getIntegerTokenValue(value->spelling, magnitude) ||
        magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + negative) {
      diagnostics.error(value->position, "integer literal is out of range");
      return std::nullopt;
    }
    if (negative && magnitude == static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1)
      return std::numeric_limits<int64_t>::min();
    int64_t signedValue = static_cast<int64_t>(magnitude);
    return negative ? -signedValue : signedValue;
  }

  bool parseShapedType(SourceType &type, std::vector<std::optional<int64_t>> &shape,
                       llvm::StringRef containerName) {
    if (!expect(TokenKind::LeftBracket, llvm::Twine("expected '[' in ") + containerName + " type"))
      return false;
    auto parsedType = parseSourceType(llvm::Twine("expected ") + containerName +
                                      " element type 'q15', 'q31', or 'f32'");
    if (!parsedType)
      return false;
    type = *parsedType;
    shape.clear();
    if (current.kind == TokenKind::Comma) {
      advance();
      do {
        auto parsedExtent =
            parseInteger(llvm::Twine("expected static ") + containerName + " extent");
        uint64_t value = 0;
        if (!parsedExtent)
          return false;
        if (getIntegerTokenValue(parsedExtent->spelling, value) ||
            value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          diagnostics.error(parsedExtent->position,
                            llvm::Twine(containerName) + " extent is out of range");
          return false;
        }
        shape.emplace_back(static_cast<int64_t>(value));
        if (current.kind != TokenKind::Comma)
          break;
        advance();
      } while (true);
    } else {
      shape.emplace_back(std::nullopt);
    }
    return expect(TokenKind::RightBracket,
                  llvm::Twine("expected ']' in ") + containerName + " type");
  }

  std::optional<ParameterAst> parseParameter() {
    auto name = parseIdentifier("expected parameter name");
    if (!name || !expect(TokenKind::Colon, "expected ':' after parameter name"))
      return std::nullopt;

    ParameterAst parameter;
    parameter.name = name->spelling.str();
    parameter.position = name->position;
    if (isIdentifier("buffer") || isIdentifier("tensor")) {
      bool isTensor = isIdentifier("tensor");
      llvm::StringRef containerName = isTensor ? "tensor" : "buffer";
      advance();
      parameter.container = isTensor ? ContainerKind::Tensor : ContainerKind::Buffer;
      if (!parseShapedType(parameter.type, parameter.shape, containerName))
        return std::nullopt;
      return parameter;
    }

    if (isIdentifier("i32") || isIdentifier("i64") || isIdentifier("bool") || isIdentifier("f32")) {
      parameter.container = ContainerKind::Scalar;
      parameter.type = *parseSourceType("expected scalar type");
      return parameter;
    }
    if (isIdentifier("complex_q15")) {
      parameter.container = ContainerKind::Scalar;
      parameter.type = SourceType::ComplexQ15;
      advance();
      return parameter;
    }

    if (!isIdentifier("constexpr")) {
      diagnostics.error(current.position,
                        "expected parameter type 'complex_q15', 'buffer[...]', 'tensor[...]', or "
                        "'constexpr[q15|q31]'");
      return std::nullopt;
    }
    advance();
    parameter.container = ContainerKind::Constexpr;
    if (!expect(TokenKind::LeftBracket, "expected '[' in constexpr type"))
      return std::nullopt;
    auto type = parseSourceType("expected constexpr element type 'q15' or 'q31'");
    if (!type || !expect(TokenKind::RightBracket, "expected ']' in constexpr type") ||
        !expect(TokenKind::Equal, "expected '=' after constexpr type") ||
        !expect(TokenKind::LeftBracket, "expected '[' before constexpr values"))
      return std::nullopt;
    parameter.type = *type;
    if (current.kind != TokenKind::RightBracket) {
      do {
        auto value = parseSignedInteger("expected integer constexpr value");
        if (!value)
          return std::nullopt;
        parameter.constantValues.push_back(*value);
        if (current.kind != TokenKind::Comma)
          break;
        advance();
      } while (true);
    }
    if (!expect(TokenKind::RightBracket, "expected ']' after constexpr values"))
      return std::nullopt;
    return parameter;
  }

  // A call is instantiated where it appears: the callee's body, with its
  // declared contracts intact, replaces the call and the arguments replace
  // the callee's parameter references. The callee is separately checked
  // against its own signature, and the arguments are checked against that
  // signature here, so the instantiated tree carries the declared result
  // type without the caller having to re-derive it.
  std::optional<BuiltinCallAst> parseCalleeInstantiation(std::optional<SourceType> policyType) {
    Token name = current;
    const KernelAst &callee = callees[calleesByName[name.spelling]];
    advance();
    advance();
    std::vector<ExpressionAst> arguments;
    if (current.kind != TokenKind::RightParen) {
      do {
        auto argument = parseIdentifier("expected a parameter or local name as a call argument");
        if (!argument)
          return std::nullopt;
        arguments.push_back(resolveOperand(*argument));
        if (current.kind != TokenKind::Comma)
          break;
        advance();
      } while (true);
    }
    if (!expect(TokenKind::RightParen, "expected ')' after call arguments"))
      return std::nullopt;
    if (arguments.size() != callee.parameters.size()) {
      diagnostics.error(name.position, llvm::Twine("'") + name.spelling + "' takes " +
                                           llvm::Twine(callee.parameters.size()) +
                                           " arguments, but " + llvm::Twine(arguments.size()) +
                                           " were given");
      return std::nullopt;
    }
    if (callee.results.size() != 1) {
      diagnostics.error(name.position,
                        llvm::Twine("'") + name.spelling +
                            "' returns two values; only single-result functions may be called");
      return std::nullopt;
    }
    if (policyType && callee.results.front().type != *policyType) {
      diagnostics.error(name.position, llvm::Twine("'") + name.spelling +
                                           "' returns a different source type than the calling "
                                           "function's result");
      return std::nullopt;
    }
    bool typedCallee =
        callee.result.kind == ReductionKind::Primitive ||
        callee.primaryResult().type == SourceType::I32 ||
        callee.primaryResult().type == SourceType::I64 ||
        callee.primaryResult().type == SourceType::Bool ||
        (callee.primaryResult().type == SourceType::F32 && isElementwiseKind(callee.result.kind)) ||
        llvm::any_of(callee.parameters, [](const ParameterAst &parameter) {
          return parameter.isScalar() && parameter.type == SourceType::F32;
        });
    if (typedCallee) {
      for (size_t index = 0; index < arguments.size(); ++index) {
        const ParameterAst &declared = callee.parameters[index];
        BuiltinCallAst checked;
        checked.kind = ReductionKind::Primitive;
        checked.primitive = PrimitiveKind::Typed;
        checked.target = declared.type;
        checked.declaredExtent =
            declared.isTensor() && declared.shape.size() == 1 ? declared.shape[0].value_or(-1) : 0;
        checked.position = name.position;
        checked.operands.push_back(std::move(arguments[index]));
        arguments[index] = ExpressionAst(std::move(checked));
      }
    }
    llvm::StringMap<const ExpressionAst *> substitution;
    for (size_t index = 0; index < arguments.size(); ++index) {
      const ParameterAst &declared = callee.parameters[index];
      substitution[declared.name] = &arguments[index];
      // An argument that names one of the caller's own parameters is checked
      // against the declaration; a local binding carries an expression whose
      // shape the instantiated tree establishes instead.
      if (!arguments[index].isParameterReference() || !caller)
        continue;
      const ParameterAst *actual = nullptr;
      for (const ParameterAst &parameter : caller->parameters)
        if (parameter.name == arguments[index].parameter)
          actual = &parameter;
      if (!actual)
        continue;
      if (actual->type != declared.type || actual->container != declared.container ||
          actual->shape != declared.shape) {
        diagnostics.error(arguments[index].position,
                          llvm::Twine("argument ") + llvm::Twine(index + 1) + " of '" +
                              name.spelling + "' does not match the declared parameter '" +
                              declared.name + "'");
        return std::nullopt;
      }
    }
    BuiltinCallAst body = instantiateCall(callee.result, substitution, name.position);
    if (!typedCallee)
      return body;
    BuiltinCallAst checked;
    checked.kind = ReductionKind::Primitive;
    checked.primitive = PrimitiveKind::Typed;
    checked.target = callee.primaryResult().type;
    checked.declaredExtent =
        callee.primaryResult().tensor && callee.primaryResult().shape.size() == 1
            ? callee.primaryResult().shape[0].value_or(-1)
            : 0;
    checked.position = name.position;
    checked.operands.push_back(ExpressionAst(std::move(body)));
    return checked;
  }

  // The optional `rounding=` then `overflow=` pair of a one-boundary member;
  // sema fills the language defaults where they are left out.
  bool parseBoundaryPolicies(BuiltinCallAst &call) {
    if (current.kind == TokenKind::Comma && next.spelling == "rounding") {
      if (!expect(TokenKind::Comma, "expected ',' before rounding policy") ||
          !expectIdentifier("rounding", "expected rounding policy") ||
          !expect(TokenKind::Equal, "expected '=' after rounding"))
        return false;
      auto rounding = parseIdentifier("expected rounding mode");
      if (!rounding)
        return false;
      call.rounding = rounding->spelling.str();
    }
    if (current.kind == TokenKind::Comma && next.spelling == "overflow") {
      if (!expect(TokenKind::Comma, "expected ',' before overflow policy") ||
          !expectIdentifier("overflow", "expected overflow policy") ||
          !expect(TokenKind::Equal, "expected '=' after overflow"))
        return false;
      auto overflow = parseIdentifier("expected overflow mode");
      if (!overflow)
        return false;
      call.destinationOverflow = overflow->spelling.str();
    }
    return true;
  }

  bool parseNamedInteger(llvm::StringRef name, int64_t &slot) {
    if (!expect(TokenKind::Comma, llvm::Twine("expected ',' before ") + name) ||
        !expectIdentifier(name, llvm::Twine("expected ") + name) ||
        !expect(TokenKind::Equal, llvm::Twine("expected '=' after ") + name))
      return false;
    auto value = parseSignedInteger(llvm::Twine("expected ") + name + " constant");
    if (!value)
      return false;
    slot = *value;
    return true;
  }

  // The default fixed contract: exact where it can be exact (an inferred
  // accumulator wide enough that no update wraps, so the mode is vacuous),
  // non-wrapping where information must be lost. Export boundaries default
  // to nearest_ties_positive, the tie rule fixed-point DSP export hardware
  // realizes natively; nearest_even stays available per call site.
  // A floating-point policy names a contract mode where a fixed one names an
  // accumulator triple, and `complex_f32` is a floating-point policy whose
  // values happen to occupy two elements each.
  static bool isFpPolicyType(SourceType type) {
    return type == SourceType::F32 || type == SourceType::ComplexF32;
  }

  static void applyDefaultFixedPolicy(BuiltinCallAst &result) {
    result.accumulatorAuto = true;
    result.rounding = "nearest_ties_positive";
    result.destinationOverflow = "saturate";
    result.updateOverflow = "wrap";
  }

  // The same rule at a fixed accumulator width. Three Q15 products bound the
  // section sum by 3*2^30 < 2^39, so wrap is vacuous at i40 here too; both
  // export boundaries lose information and take the export-default tie rule.
  static void applyDefaultSosPolicy(BuiltinCallAst &result) {
    result.accumulatorAuto = true;
    result.accumulatorWidth = 40;
    result.updateOverflow = "wrap";
    result.stateRounding = "nearest_ties_positive";
    result.stateOverflow = "saturate";
    result.rounding = "nearest_ties_positive";
    result.destinationOverflow = "saturate";
  }

  bool parseFixedPolicy(BuiltinCallAst &result) {
    // The product selection leads the spelled policy; left out, it is the
    // exact full product every profile starts from.
    if (isIdentifier("product")) {
      advance();
      if (!expect(TokenKind::Equal, "expected '=' after product"))
        return false;
      auto product = parseIdentifier("expected product selection 'full' or 'raw_high'");
      if (!product || !expect(TokenKind::Comma, "expected ',' before accumulator policy"))
        return false;
      result.product = product->spelling.str();
    }
    if (!parseAccumulatorPolicy(result) ||
        !expect(TokenKind::Comma, "expected ',' before rounding policy") ||
        !expectIdentifier("rounding", "expected rounding policy") ||
        !expect(TokenKind::Equal, "expected '=' after rounding"))
      return false;
    auto rounding = parseIdentifier("expected rounding mode");
    if (!rounding || !expect(TokenKind::Comma, "expected ',' before destination overflow policy") ||
        !expectIdentifier("overflow", "expected destination overflow policy") ||
        !expect(TokenKind::Equal, "expected '=' after overflow"))
      return false;
    result.rounding = rounding->spelling.str();
    auto destinationOverflow = parseIdentifier("expected destination overflow mode");
    if (!destinationOverflow)
      return false;
    result.destinationOverflow = destinationOverflow->spelling.str();
    return true;
  }

  bool parseAccumulatorPolicy(BuiltinCallAst &result) {
    if (!expectIdentifier("accumulator", "expected accumulator policy") ||
        !expect(TokenKind::Equal, "expected '=' after accumulator") ||
        !expectIdentifier("exact", "only exact accumulator semantics are currently supported") ||
        !expect(TokenKind::LeftBracket, "expected '[' after exact"))
      return false;
    auto width = parseInteger("expected exact accumulator width");
    if (!width || !expect(TokenKind::Comma, "expected ',' after accumulator width"))
      return false;
    if (getIntegerTokenValue(width->spelling, result.accumulatorWidth)) {
      diagnostics.error(width->position, "accumulator width is out of range");
      return false;
    }
    auto updateOverflow = parseIdentifier("expected accumulator update overflow policy");
    if (!updateOverflow ||
        !expect(TokenKind::RightBracket, "expected ']' after accumulator policy"))
      return false;
    result.updateOverflow = updateOverflow->spelling.str();
    return true;
  }

  bool parseFixedSosPolicy(BuiltinCallAst &result) {
    if (!parseAccumulatorPolicy(result) ||
        !expect(TokenKind::Comma, "expected ',' before state rounding policy") ||
        !expectIdentifier("state_rounding", "expected state rounding policy") ||
        !expect(TokenKind::Equal, "expected '=' after state_rounding"))
      return false;
    auto stateRounding = parseIdentifier("expected state rounding mode");
    if (!stateRounding ||
        !expect(TokenKind::Comma, "expected ',' before state destination overflow policy") ||
        !expectIdentifier("state_overflow", "expected state destination overflow policy") ||
        !expect(TokenKind::Equal, "expected '=' after state_overflow"))
      return false;
    result.stateRounding = stateRounding->spelling.str();
    auto stateOverflow = parseIdentifier("expected state destination overflow mode");
    if (!stateOverflow || !expect(TokenKind::Comma, "expected ',' before output rounding policy") ||
        !expectIdentifier("output_rounding", "expected output rounding policy") ||
        !expect(TokenKind::Equal, "expected '=' after output_rounding"))
      return false;
    result.stateOverflow = stateOverflow->spelling.str();
    auto outputRounding = parseIdentifier("expected output rounding mode");
    if (!outputRounding ||
        !expect(TokenKind::Comma, "expected ',' before output destination overflow policy") ||
        !expectIdentifier("output_overflow", "expected output destination overflow policy") ||
        !expect(TokenKind::Equal, "expected '=' after output_overflow"))
      return false;
    result.rounding = outputRounding->spelling.str();
    auto outputOverflow = parseIdentifier("expected output destination overflow mode");
    if (!outputOverflow)
      return false;
    result.destinationOverflow = outputOverflow->spelling.str();
    return true;
  }

  struct Binding {
    std::string name;
    ExpressionAst expression;
    SourcePosition position;
    bool consumed = false;
    std::string sourceName;
    bool readOnly = false;
  };

  bool isLoopStart() const { return isIdentifier("for") && next.kind == TokenKind::Identifier; }
  bool isStatementStart() const {
    return current.kind == TokenKind::Identifier &&
           (next.kind == TokenKind::Equal || (genericExpressions && isLoopStart()));
  }
  const ParameterAst *findParameter(StringRef name) const {
    auto found = llvm::find_if(
        caller->parameters, [&](const ParameterAst &parameter) { return parameter.name == name; });
    return found == caller->parameters.end() ? nullptr : &*found;
  }

  // One assignment or loop appended to `owner`. A reassigned local keeps its
  // name; Sema requires it to keep its type.
  bool parseStatement(BuiltinCallAst &owner, SourceType policyType) {
    if (isLoopStart())
      return parseLoop(owner, policyType);
    Token name = current;
    auto found = bindingsByName.find(name.spelling);
    if (found != bindingsByName.end() && bindings[found->second].readOnly) {
      diagnostics.error(name.position,
                        llvm::Twine("loop variable '") + name.spelling + "' is read-only");
      return false;
    }
    if (findParameter(name.spelling) || (found != bindingsByName.end() && !genericExpressions)) {
      diagnostics.error(name.position, llvm::Twine("local '") + name.spelling +
                                           "' collides with an existing name");
      return false;
    }
    advance();
    advance();
    std::optional<BuiltinCallAst> bound = parseStatementExpression(policyType);
    if (!bound)
      return false;
    if (found == bindingsByName.end()) {
      scopedOut.erase(name.spelling);
      bindingsByName[name.spelling] = bindings.size();
      bindings.push_back(Binding{genericExpressions ? ("$local" + std::to_string(nextLocal++))
                                                    : name.spelling.str(),
                                 ExpressionAst(*bound), name.position, false, name.spelling.str()});
      found = bindingsByName.find(name.spelling);
    }
    owner.bindingNames.push_back(bindings[found->second].name);
    owner.operands.push_back(ExpressionAst(std::move(*bound)));
    return true;
  }

  // `for v in range([start,] stop[, step]):` then the statements indented
  // past the `for`. Names first bound in the body do not outlive it.
  bool parseLoop(BuiltinCallAst &owner, SourceType policyType) {
    Token keyword = current;
    advance();
    Token variable = current;
    advance();
    if (bindingsByName.contains(variable.spelling) || findParameter(variable.spelling)) {
      diagnostics.error(variable.position, llvm::Twine("loop variable '") + variable.spelling +
                                               "' collides with an existing name");
      return false;
    }
    if (!expectIdentifier("in", "expected 'in' after the loop variable") ||
        !expectIdentifier("range", "expected 'range(...)' after 'in'") ||
        !expect(TokenKind::LeftParen, "expected '(' after 'range'"))
      return false;
    std::vector<ExpressionAst> bounds;
    do {
      if (!bounds.empty())
        advance();
      std::optional<Operand> bound = parseLogicalOr();
      if (!bound)
        return false;
      bounds.push_back(std::move(*bound->expression));
    } while (current.kind == TokenKind::Comma && bounds.size() < 3);
    if (!expect(TokenKind::RightParen, "range takes at most three arguments; expected ')'") ||
        !expect(TokenKind::Colon, "expected ':' after the range"))
      return false;

    BuiltinCallAst loop = makePrimitive(PrimitiveKind::Loop, {}, keyword.position);
    if (bounds.size() == 3) {
      const BuiltinCallAst *step = bounds.back().call.get();
      int64_t value = 0;
      bool literal = step && step->kind == ReductionKind::Primitive &&
                     step->primitive == PrimitiveKind::Literal &&
                     !StringRef(step->spelling).getAsInteger(10, value);
      if (!literal || value == 0) {
        diagnostics.error(bounds.back().position, "range step must be a nonzero integer literal");
        return false;
      }
      loop.step = value;
      bounds.pop_back();
    }
    if (bounds.size() == 1) {
      BuiltinCallAst zero = makePrimitive(PrimitiveKind::Literal, {}, keyword.position);
      zero.spelling = "0";
      loop.operands.push_back(ExpressionAst(std::move(zero)));
    }
    for (ExpressionAst &bound : bounds)
      loop.operands.push_back(std::move(bound));
    std::string induction = "$local" + std::to_string(nextLocal++);
    loop.bindingNames = {induction, ""};

    llvm::StringMap<size_t> outer = bindingsByName;
    bindingsByName[variable.spelling] = bindings.size();
    bindings.push_back(Binding{induction, ExpressionAst(induction, variable.position),
                               variable.position, true, variable.spelling.str(), true});
    unsigned column = current.position.column;
    if (current.kind == TokenKind::Eof || column <= keyword.position.column) {
      diagnostics.error(current.position, "expected an indented loop body");
      return false;
    }
    while (current.kind != TokenKind::Eof && current.position.column == column) {
      if (!isStatementStart()) {
        diagnostics.error(current.position, "expected an assignment or a nested loop");
        return false;
      }
      if (!parseStatement(loop, policyType))
        return false;
    }
    if (current.kind != TokenKind::Eof && current.position.column > keyword.position.column) {
      diagnostics.error(current.position, "inconsistent indentation in the loop body");
      return false;
    }
    for (const auto &entry : bindingsByName)
      if (!outer.contains(entry.getKey()))
        scopedOut.insert(entry.getKey());
    bindingsByName = std::move(outer);
    owner.bindingNames.push_back("");
    owner.operands.push_back(ExpressionAst(std::move(loop)));
    return true;
  }

  BuiltinCallAst bindLocals(BuiltinCallAst scope, BuiltinCallAst result) {
    std::function<bool(const BuiltinCallAst &)> isLegacyTree = [&](const BuiltinCallAst &call) {
      if (call.kind == ReductionKind::Primitive && call.primitive != PrimitiveKind::Identity)
        return false;
      return llvm::all_of(call.operands, [&](const ExpressionAst &operand) {
        return operand.isParameterReference() || isLegacyTree(*operand.call);
      });
    };
    llvm::StringSet<> assigned;
    bool legacy = isLegacyTree(result) &&
                  llvm::all_of(llvm::zip(scope.bindingNames, scope.operands), [&](auto statement) {
                    return assigned.insert(std::get<0>(statement)).second &&
                           isLegacyTree(*std::get<1>(statement).call);
                  });
    if (legacy) {
      std::vector<ExpressionAst> values;
      values.reserve(scope.operands.size());
      llvm::StringMap<const ExpressionAst *> names;
      for (auto [name, expression] : llvm::zip(scope.bindingNames, scope.operands)) {
        ExpressionAst value = instantiateExpression(expression, names, std::nullopt);
        if (value.call && value.call->kind == ReductionKind::Primitive &&
            value.call->primitive == PrimitiveKind::Identity) {
          ExpressionAst operand = std::move(value.call->operands.front());
          value = std::move(operand);
        }
        values.push_back(std::move(value));
        names[name] = &values.back();
      }
      result = instantiateCall(result, names, std::nullopt);
      if (result.kind == ReductionKind::Primitive && result.primitive == PrimitiveKind::Identity &&
          result.operands.front().call) {
        BuiltinCallAst nested = std::move(*result.operands.front().call);
        result = std::move(nested);
      }
      return result;
    }
    scope.position = result.position;
    scope.operands.push_back(ExpressionAst(std::move(result)));
    return scope;
  }

  // Every reference to a local instantiates a fresh deep copy of the bound
  // call tree. The Ondrix tensor operations the copies produce are Pure, so
  // the pipeline's canonicalize/cse collapses them back to one evaluation.
  ExpressionAst resolveOperand(const Token &token) {
    auto binding = bindingsByName.find(token.spelling);
    if (binding == bindingsByName.end()) {
      if (scopedOut.contains(token.spelling) && !scopedOutUse)
        scopedOutUse = token;
      return ExpressionAst(token.spelling.str(), token.position);
    }
    Binding &bound = bindings[binding->second];
    bound.consumed = true;
    if (genericExpressions)
      return ExpressionAst(bound.name, token.position);
    return ExpressionAst(bound.expression);
  }

  bool genericExpressions = false;
  unsigned nextLocal = 0;
  // Names bound only inside a finished loop body, and the first later read.
  llvm::StringSet<> scopedOut;
  std::optional<Token> scopedOutUse;
  Lexer &lexer;
  Diagnostics &diagnostics;
  Token current;
  Token next;
  std::vector<Binding> bindings;
  llvm::StringMap<size_t> bindingsByName;
  std::vector<KernelAst> callees;
  llvm::StringMap<size_t> calleesByName;
  const KernelAst *caller = nullptr;
};

struct CheckedKernel {
  KernelAst ast;
  std::optional<ondsp::OverflowMode> updateOverflow = std::nullopt;
  std::optional<ondsp::RoundingMode> rounding = std::nullopt;
  std::optional<ondsp::OverflowMode> destinationOverflow = std::nullopt;
  std::optional<ondsp::FpContractMode> fpContract = std::nullopt;
  std::optional<ondsp::RoundingMode> stateRounding = std::nullopt;
  std::optional<ondsp::OverflowMode> stateOverflow = std::nullopt;
  std::optional<ondsp::RoundingMode> inputRounding = std::nullopt;
  // Distinct from inputRounding: matmul and lms round a PRODUCT, not an
  // input, and one field for both boundaries made a later reader assume the
  // magnitude component pre-shift was already covered when it was not.
  std::optional<ondsp::RoundingMode> productRounding = std::nullopt;
};

static std::optional<ondsp::OverflowMode> parseOverflow(llvm::StringRef value) {
  if (value == "wrap")
    return ondsp::OverflowMode::Wrap;
  if (value == "saturate")
    return ondsp::OverflowMode::Saturate;
  return std::nullopt;
}

static std::optional<ondsp::RoundingMode> parseRounding(llvm::StringRef value) {
  if (value == "toward_negative")
    return ondsp::RoundingMode::TowardNegative;
  if (value == "toward_zero")
    return ondsp::RoundingMode::TowardZero;
  if (value == "nearest_even")
    return ondsp::RoundingMode::NearestEven;
  if (value == "nearest_ties_positive")
    return ondsp::RoundingMode::NearestTiesPositive;
  return std::nullopt;
}

// The rounding modes every accumulator-export contract in the language covers
// (dot/FIR-family export and the SOS state/output boundaries carry per-op
// discriminating object evidence for all four). A newly declared dialect mode
// is opted into per builtin, together with the operation contract and its
// differential evidence; it never reaches a binding just because the enum
// grew a case.
static bool isDeclaredExportRounding(ondsp::RoundingMode mode) {
  return mode == ondsp::RoundingMode::TowardNegative || mode == ondsp::RoundingMode::TowardZero ||
         mode == ondsp::RoundingMode::NearestEven ||
         mode == ondsp::RoundingMode::NearestTiesPositive;
}

// The two executable packed complex profiles. A container holds two
// components side by side, so the source type fixes the component width and
// the layout together, and the accumulator carries the exact product frac.
static bool isPackedComplexType(SourceType type) {
  return type == SourceType::ComplexQ15 || type == SourceType::ComplexQ31;
}

static unsigned getComplexComponentWidth(SourceType type) {
  assert(isPackedComplexType(type) && "a packed profile is the caller's precondition");
  return type == SourceType::ComplexQ15 ? 16 : 32;
}

// A component term is a DIFFERENCE of two products, so no width makes the
// update vacuous and every call site declares its carrier. 32 is the one a
// packed complex target reads back; 40 and 64 are the Q15 and Q31 reduction
// families' own.
static bool isDeclaredComplexAccumulatorWidth(SourceType type, uint64_t width) {
  assert(isPackedComplexType(type) && "a packed profile is the caller's precondition");
  return type == SourceType::ComplexQ15 ? width == 32 || width == 40 : width == 64;
}

static llvm::StringRef describeComplexAccumulatorWidths(SourceType type) {
  assert(isPackedComplexType(type) && "a packed profile is the caller's precondition");
  return type == SourceType::ComplexQ15 ? "32 or 40" : "64";
}

static ondsp::ComplexLayout getPackedComplexLayout(SourceType type) {
  assert(isPackedComplexType(type) && "a packed profile is the caller's precondition");
  return type == SourceType::ComplexQ15 ? ondsp::ComplexLayout::PackedI16ImagHiRealLo
                                        : ondsp::ComplexLayout::PackedI32ImagHiRealLo;
}

static std::optional<ondsp::FpContractMode> parseFpContract(llvm::StringRef value) {
  if (value == "off")
    return ondsp::FpContractMode::Off;
  if (value == "fma")
    return ondsp::FpContractMode::Fma;
  if (value == "fast")
    return ondsp::FpContractMode::Fast;
  return std::nullopt;
}

static unsigned inferQ15FullAccumulatorWidth(uint64_t productCount) {
  llvm::APInt maximumMagnitude(128, productCount);
  maximumMagnitude <<= 30;
  return std::max(32u, maximumMagnitude.getActiveBits() + 1);
}

static std::optional<llvm::StringRef> getParameterOperand(const BuiltinCallAst &call,
                                                          unsigned index) {
  if (index >= call.operands.size() || !call.operands[index].isParameterReference())
    return std::nullopt;
  return call.operands[index].parameter;
}

static std::optional<CheckedKernel> checkKernel(KernelAst ast, Diagnostics &diagnostics) {
  if (isDesignKind(ast.result.kind)) {
    diagnostics.error(ast.result.position,
                      llvm::Twine(describeDesignKind(ast.result.kind)) +
                          " is a design expression; it is consumed by fir_filter coefficients");
    return std::nullopt;
  }
  bool hasThreeOperands = ast.result.kind == ReductionKind::Butterfly ||
                          ast.result.kind == ReductionKind::FirStream ||
                          ast.result.kind == ReductionKind::Lms;
  bool hasFourOperands =
      ast.result.kind == ReductionKind::SosDf2Fixed || ast.result.kind == ReductionKind::SosTdf2;
  size_t expectedOperandCount = isUnaryKind(ast.result.kind) ? 1
                                : hasFourOperands            ? 4
                                : hasThreeOperands           ? 3
                                                             : 2;
  if (ast.result.kind != ReductionKind::Primitive &&
      ast.result.operands.size() != expectedOperandCount) {
    diagnostics.error(ast.result.position, "builtin operand count does not match its contract");
    return std::nullopt;
  }
  if (!isComposableKernel(ast.result.kind, ast.primaryResult().type) &&
      llvm::any_of(ast.result.operands,
                   [](const ExpressionAst &operand) { return !operand.isParameterReference(); })) {
    diagnostics.error(ast.result.position,
                      "nested calls are currently supported only by the FFT-family and "
                      "elementwise builtins");
    return std::nullopt;
  }
  size_t expectedParameterCount = isUnaryKind(ast.result.kind) ? 1
                                  : hasFourOperands            ? 4
                                  : hasThreeOperands           ? 3
                                                               : 2;
  // FFT-composable kernels take however many parameters their expression
  // tree consumes; the exactly-once accounting below replaces the count.
  if (!isComposableKernel(ast.result.kind, ast.primaryResult().type) &&
      ast.parameters.size() != expectedParameterCount) {
    diagnostics.error(ast.position,
                      isUnaryKind(ast.result.kind)
                          ? "unary DSP kernels require exactly one parameter"
                      : hasFourOperands  ? "second-order-section kernels require exactly four "
                                           "parameters"
                      : hasThreeOperands ? "butterfly, fir_stream, and lms kernels require exactly "
                                           "three parameters"
                                         : "binary DSP kernels require exactly two parameters");
    return std::nullopt;
  }
  auto verifyPositiveShape = [&](llvm::ArrayRef<std::optional<int64_t>> shape,
                                 llvm::StringRef description, SourcePosition position) {
    if (llvm::any_of(shape, [](std::optional<int64_t> extent) { return extent && *extent <= 0; })) {
      diagnostics.error(position,
                        llvm::Twine("static ") + description + " tensor extents must be positive");
      return false;
    }
    return true;
  };
  if (!verifyPositiveShape(ast.primaryResult().shape, "result", ast.position) ||
      (ast.results.size() > 1 &&
       !verifyPositiveShape(ast.results[1].shape, "second result", ast.position)))
    return std::nullopt;

  llvm::StringSet<> parameterNames;
  llvm::DenseMap<llvm::StringRef, const ParameterAst *> parametersByName;
  const ParameterAst *lhsParameter = nullptr;
  const ParameterAst *rhsParameter = nullptr;
  const ParameterAst *thirdParameter = nullptr;
  const ParameterAst *fourthParameter = nullptr;
  std::optional<llvm::StringRef> lhsName = getParameterOperand(ast.result, 0);
  std::optional<llvm::StringRef> rhsName = getParameterOperand(ast.result, 1);
  std::optional<llvm::StringRef> thirdName = getParameterOperand(ast.result, 2);
  std::optional<llvm::StringRef> fourthName = getParameterOperand(ast.result, 3);
  if (ast.primaryResult().type == SourceType::U8 &&
      ast.result.kind != ReductionKind::ViterbiDecode) {
    diagnostics.error(ast.result.position, "u8 is currently only the result of viterbi_decode");
    return std::nullopt;
  }
  for (const ParameterAst &parameter : ast.parameters) {
    if (!parameterNames.insert(parameter.name).second) {
      diagnostics.error(parameter.position,
                        llvm::Twine("duplicate parameter '") + parameter.name + "'");
      return std::nullopt;
    }
    if (!isComposableKernel(ast.result.kind, ast.primaryResult().type) &&
        parameter.type != ast.primaryResult().type) {
      diagnostics.error(parameter.position,
                        "parameter element types must match the kernel result type");
      return std::nullopt;
    }
    if (!verifyPositiveShape(parameter.shape, "parameter", parameter.position))
      return std::nullopt;
    if (parameter.type == SourceType::U8) {
      diagnostics.error(parameter.position, "u8 is currently only the result of viterbi_decode");
      return std::nullopt;
    }
    if ((parameter.type == SourceType::I32 || parameter.type == SourceType::I64 ||
         parameter.type == SourceType::Bool) &&
        !parameter.isScalar()) {
      diagnostics.error(parameter.position, "i32, i64 and bool parameters must be scalar");
      return std::nullopt;
    }
    parametersByName.insert({parameter.name, &parameter});
    if (lhsName && parameter.name == *lhsName)
      lhsParameter = &parameter;
    if (rhsName && parameter.name == *rhsName)
      rhsParameter = &parameter;
    if (thirdName && parameter.name == *thirdName)
      thirdParameter = &parameter;
    if (fourthName && parameter.name == *fourthName)
      fourthParameter = &parameter;
  }
  if (!isComposableKernel(ast.result.kind, ast.primaryResult().type) &&
      (!lhsName || !parameterNames.contains(*lhsName))) {
    diagnostics.error(ast.result.position, llvm::Twine("unknown builtin operand '") +
                                               (lhsName ? *lhsName : "<nested call>") + "'");
    return std::nullopt;
  }
  if (!isUnaryKind(ast.result.kind) &&
      !isComposableKernel(ast.result.kind, ast.primaryResult().type) &&
      (!rhsName || !parameterNames.contains(*rhsName))) {
    diagnostics.error(ast.result.position, llvm::Twine("unknown reduction operand '") +
                                               (rhsName ? *rhsName : "<nested call>") + "'");
    return std::nullopt;
  }
  if (hasThreeOperands && (!thirdName || !parameterNames.contains(*thirdName))) {
    diagnostics.error(ast.result.position, llvm::Twine("unknown third builtin operand '") +
                                               (thirdName ? *thirdName : "<nested call>") + "'");
    return std::nullopt;
  }
  if (hasFourOperands && (!thirdName || !fourthName || !parameterNames.contains(*thirdName) ||
                          !parameterNames.contains(*fourthName))) {
    diagnostics.error(ast.result.position, "unknown second-order-section builtin operand");
    return std::nullopt;
  }

  unsigned constexprCount = llvm::count_if(
      ast.parameters, [](const ParameterAst &parameter) { return parameter.isConstexpr(); });
  if (ast.result.kind == ReductionKind::Butterfly) {
    if (ast.results.size() != 2 || ast.primaryResult().tensor || ast.results[1].tensor ||
        ast.primaryResult().type != SourceType::ComplexQ15 ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isScalar(); })) {
      diagnostics.error(ast.result.position,
                        "butterfly requires three complex_q15 scalar parameters and two "
                        "complex_q15 results");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, std::nullopt};
  }
  if (ast.result.kind == ReductionKind::CxDot) {
    SourceType complexType = ast.primaryResult().type;
    if (complexType == SourceType::ComplexF32) {
      // The operation carries the interleaved f32 profile; the SOURCE has no
      // spelling for a complex f32 scalar, because a function returns one
      // value and this result is two. Naming that is a language decision, so
      // it is refused here rather than deferred silently.
      diagnostics.error(ast.result.position,
                        "cx_dot has no complex_f32 spelling: a complex f32 result is two values "
                        "and a kernel returns one; use cx_fir_filter, whose result is a tensor");
      return std::nullopt;
    }
    if (ast.results.size() != 1 || ast.primaryResult().tensor ||
        !isPackedComplexType(complexType) || !lhsParameter || !rhsParameter ||
        !lhsParameter->isBuffer() || !rhsParameter->isBuffer() ||
        !hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 1)) {
      diagnostics.error(ast.result.position,
                        "cx_dot requires two rank-1 complex_q15 or complex_q31 buffer parameters "
                        "and one matching complex scalar result");
      return std::nullopt;
    }
    const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &rhsExtent = getRankOneExtent(rhsParameter->shape);
    if (lhsExtent && rhsExtent && *lhsExtent != *rhsExtent) {
      diagnostics.error(ast.result.position, "cx_dot operands must have equal static extents");
      return std::nullopt;
    }
    if (ast.result.accumulatorAuto ||
        !isDeclaredComplexAccumulatorWidth(complexType, ast.result.accumulatorWidth)) {
      diagnostics.error(ast.result.position,
                        llvm::Twine("cx_dot requires an explicit exact accumulator of width ") +
                            describeComplexAccumulatorWidths(complexType));
      return std::nullopt;
    }
    auto updateOverflow = parseOverflow(ast.result.updateOverflow);
    if (!updateOverflow) {
      diagnostics.error(ast.result.position, llvm::Twine("unsupported update overflow mode '") +
                                                 ast.result.updateOverflow + "'");
      return std::nullopt;
    }
    auto rounding = parseRounding(ast.result.rounding);
    if (!rounding || !isDeclaredExportRounding(*rounding)) {
      diagnostics.error(ast.result.position,
                        "export rounding must be nearest_even, nearest_ties_positive, "
                        "toward_negative, or toward_zero");
      return std::nullopt;
    }
    auto destinationOverflow = parseOverflow(ast.result.destinationOverflow);
    if (!destinationOverflow) {
      diagnostics.error(ast.result.position,
                        llvm::Twine("unsupported destination overflow mode '") +
                            ast.result.destinationOverflow + "'");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), *updateOverflow, *rounding, *destinationOverflow,
                         std::nullopt};
  }
  if (ast.result.kind == ReductionKind::CxFirFilter) {
    SourceType complexType = ast.primaryResult().type;
    bool interleavedFp = complexType == SourceType::ComplexF32;
    if (ast.results.size() != 1 || !ast.primaryResult().tensor ||
        (!isPackedComplexType(complexType) && !interleavedFp) || !lhsParameter || !rhsParameter ||
        !lhsParameter->isTensor() || !rhsParameter->isTensor() ||
        !hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 1) ||
        !hasRank(ast.primaryResult().shape, 1)) {
      diagnostics.error(ast.result.position,
                        "cx_fir_filter requires two rank-1 complex_q15, complex_q31 or "
                        "complex_f32 tensor parameters and one matching complex tensor result");
      return std::nullopt;
    }
    if (ast.result.boundary != "valid") {
      diagnostics.error(ast.result.position, "cx_fir_filter currently supports boundary=valid");
      return std::nullopt;
    }
    const std::optional<int64_t> &inputExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &tapExtent = getRankOneExtent(rhsParameter->shape);
    const std::optional<int64_t> &outputExtent = getRankOneExtent(ast.primaryResult().shape);
    if (!inputExtent || !tapExtent || !outputExtent) {
      diagnostics.error(ast.result.position, "cx_fir_filter requires static extents");
      return std::nullopt;
    }
    if (*inputExtent < *tapExtent || *outputExtent != *inputExtent - *tapExtent + 1) {
      diagnostics.error(ast.result.position,
                        "valid-boundary cx_fir_filter requires input extent >= tap count and "
                        "result extent = input extent - tap count + 1");
      return std::nullopt;
    }
    if (interleavedFp) {
      // The format is the carrier, so the window declares a contract mode
      // where the packed widths declare an accumulator triple.
      std::optional<ondsp::FpContractMode> contract = parseFpContract(ast.result.fpContract);
      if (!contract) {
        diagnostics.error(ast.result.position,
                          "an interleaved f32 cx_fir_filter requires contract = off, fma, or fast");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
    }
    // The per-sample contract is the one cx_dot declares; the window only
    // decides how many times it runs.
    if (ast.result.accumulatorAuto ||
        !isDeclaredComplexAccumulatorWidth(complexType, ast.result.accumulatorWidth)) {
      diagnostics.error(
          ast.result.position,
          llvm::Twine("cx_fir_filter requires an explicit exact accumulator of width ") +
              describeComplexAccumulatorWidths(complexType));
      return std::nullopt;
    }
    auto updateOverflow = parseOverflow(ast.result.updateOverflow);
    auto rounding = parseRounding(ast.result.rounding);
    auto destinationOverflow = parseOverflow(ast.result.destinationOverflow);
    if (!updateOverflow || !destinationOverflow) {
      diagnostics.error(ast.result.position, "unsupported cx_fir_filter overflow mode");
      return std::nullopt;
    }
    if (!rounding || !isDeclaredExportRounding(*rounding)) {
      diagnostics.error(ast.result.position,
                        "export rounding must be nearest_even, nearest_ties_positive, "
                        "toward_negative, or toward_zero");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), *updateOverflow, *rounding, *destinationOverflow,
                         std::nullopt};
  }
  if (ast.result.kind == ReductionKind::FirStream) {
    bool isQ31 = ast.primaryResult().type == SourceType::Q31;
    if (ast.results.size() != 2 || !ast.primaryResult().tensor || !ast.results[1].tensor ||
        (ast.primaryResult().type != SourceType::Q15 && !isQ31) || !lhsParameter || !rhsParameter ||
        !thirdParameter || llvm::any_of(ast.parameters, [](const ParameterAst &parameter) {
          return !parameter.isTensor();
        })) {
      diagnostics.error(ast.result.position, "fir_stream requires three Q15 or Q31 tensor "
                                             "parameters and two matching tensor results");
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 1) ||
        !hasRank(thirdParameter->shape, 1) || !hasRank(ast.primaryResult().shape, 1) ||
        !hasRank(ast.results[1].shape, 1)) {
      diagnostics.error(ast.result.position, "fir_stream currently requires rank-1 tensors");
      return std::nullopt;
    }
    const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &rhsExtent = getRankOneExtent(rhsParameter->shape);
    const std::optional<int64_t> &stateExtent = getRankOneExtent(thirdParameter->shape);
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    const std::optional<int64_t> &nextStateExtent = getRankOneExtent(ast.results[1].shape);
    if (!rhsExtent || !stateExtent || !nextStateExtent) {
      diagnostics.error(ast.result.position,
                        "fir_stream currently requires static coefficient, state, and next-state "
                        "extents");
      return std::nullopt;
    }
    int64_t coefficientExtent = *rhsExtent;
    int64_t expectedStateExtent = coefficientExtent - 1;
    if (*stateExtent != expectedStateExtent || *nextStateExtent != expectedStateExtent)
      diagnostics.error(ast.result.position,
                        "fir_stream state and next-state extents must equal coefficients - 1");
    else if (lhsExtent.has_value() != resultExtent.has_value())
      diagnostics.error(ast.result.position,
                        "fir_stream input and output must both be static or both be dynamic");
    else if (lhsExtent && *resultExtent != *lhsExtent)
      diagnostics.error(ast.result.position,
                        "fir_stream output extent must equal the input chunk extent");
    else if (ast.result.accumulatorAuto) {
      // Two Q31 products already leave i64, so no inferred width makes the
      // update mode vacuous there; the Q31 stream spells its whole contract.
      if (isQ31) {
        diagnostics.error(ast.result.position, "a Q31 fir_stream requires an explicit accumulator, "
                                               "rounding, and overflow policy");
        return std::nullopt;
      }
      ast.result.accumulatorWidth =
          inferQ15FullAccumulatorWidth(static_cast<uint64_t>(coefficientExtent));
      return CheckedKernel{std::move(ast), ondsp::OverflowMode::Wrap,
                           ondsp::RoundingMode::NearestTiesPositive, ondsp::OverflowMode::Saturate,
                           std::nullopt};
    } else {
      if (ast.result.accumulatorWidth != (isQ31 ? 64u : 40u)) {
        diagnostics.error(ast.result.position,
                          isQ31 ? "the executable Q31 profile requires exact accumulator width 64"
                                : "the executable Q15 profile requires exact accumulator width 40");
        return std::nullopt;
      }
      auto updateOverflow = parseOverflow(ast.result.updateOverflow);
      auto rounding = parseRounding(ast.result.rounding);
      auto destinationOverflow = parseOverflow(ast.result.destinationOverflow);
      if (!updateOverflow || !rounding || !destinationOverflow) {
        diagnostics.error(ast.result.position, "fir_stream contains an unsupported numeric policy");
        return std::nullopt;
      }
      if (!isDeclaredExportRounding(*rounding)) {
        diagnostics.error(ast.result.position,
                          "export rounding must be nearest_even, nearest_ties_positive, "
                          "toward_negative, or toward_zero");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast), *updateOverflow, *rounding, *destinationOverflow,
                           std::nullopt};
    }
    return std::nullopt;
  }
  if (ast.result.kind == ReductionKind::SosDf2Fixed) {
    bool isQ31 = ast.primaryResult().type == SourceType::Q31;
    if (ast.results.size() != 2 || !ast.primaryResult().tensor || !ast.results[1].tensor ||
        (ast.primaryResult().type != SourceType::Q15 && !isQ31) || !lhsParameter || !rhsParameter ||
        !thirdParameter || !fourthParameter || constexprCount != 0 ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isTensor(); })) {
      diagnostics.error(ast.result.position, "sos_df2_fixed requires four Q15 or Q31 tensor "
                                             "parameters and two matching tensor results");
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 2) ||
        !hasRank(thirdParameter->shape, 1) || !hasRank(fourthParameter->shape, 2) ||
        !hasRank(ast.primaryResult().shape, 1) || !hasRank(ast.results[1].shape, 2)) {
      diagnostics.error(ast.result.position, "sos_df2_fixed requires input/output rank 1, "
                                             "coefficients/state rank 2, and scales rank 1");
      return std::nullopt;
    }
    const std::optional<int64_t> &inputExtent = lhsParameter->shape[0];
    const std::optional<int64_t> &outputExtent = ast.primaryResult().shape[0];
    if (inputExtent.has_value() != outputExtent.has_value() ||
        (inputExtent && *inputExtent != *outputExtent)) {
      diagnostics.error(ast.result.position,
                        "sos_df2_fixed input and output chunk extents must match");
      return std::nullopt;
    }
    if (rhsParameter->shape[0] != std::optional<int64_t>(1) ||
        rhsParameter->shape[1] != std::optional<int64_t>(5) ||
        thirdParameter->shape[0] != std::optional<int64_t>(1) ||
        fourthParameter->shape[0] != std::optional<int64_t>(1) ||
        fourthParameter->shape[1] != std::optional<int64_t>(2) ||
        ast.results[1].shape[0] != std::optional<int64_t>(1) ||
        ast.results[1].shape[1] != std::optional<int64_t>(2)) {
      diagnostics.error(
          ast.result.position,
          "sos_df2_fixed currently requires coefficients [1,5], scales [1], and state [1,2]");
      return std::nullopt;
    }
    // Three Q31 products leave i64, so the Q15 default's vacuous wrap has no
    // Q31 counterpart; the Q31 section spells its whole contract.
    if (isQ31 && ast.result.accumulatorAuto) {
      diagnostics.error(ast.result.position,
                        "a Q31 sos_df2_fixed requires an explicit accumulator, "
                        "state, and output policy");
      return std::nullopt;
    }
    if (ast.result.accumulatorWidth != (isQ31 ? 64u : 40u)) {
      diagnostics.error(ast.result.position,
                        isQ31
                            ? "the executable Q31 SOS profile requires exact accumulator width 64"
                            : "the executable Q15 SOS profile requires exact accumulator width 40");
      return std::nullopt;
    }
    auto updateOverflow = parseOverflow(ast.result.updateOverflow);
    auto stateRounding = parseRounding(ast.result.stateRounding);
    auto stateOverflow = parseOverflow(ast.result.stateOverflow);
    auto outputRounding = parseRounding(ast.result.rounding);
    auto outputOverflow = parseOverflow(ast.result.destinationOverflow);
    if (!updateOverflow || !stateRounding || !stateOverflow || !outputRounding || !outputOverflow) {
      diagnostics.error(ast.result.position,
                        "sos_df2_fixed contains an unsupported numeric policy");
      return std::nullopt;
    }
    if (!isDeclaredExportRounding(*stateRounding)) {
      diagnostics.error(ast.result.position,
                        "sos_df2_fixed state_rounding must be nearest_even, nearest_ties_positive, "
                        "toward_negative, or toward_zero");
      return std::nullopt;
    }
    if (!isDeclaredExportRounding(*outputRounding)) {
      diagnostics.error(
          ast.result.position,
          "sos_df2_fixed output_rounding must be nearest_even, nearest_ties_positive, "
          "toward_negative, or toward_zero");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), *updateOverflow, *outputRounding, *outputOverflow,
                         std::nullopt,   *stateRounding,  *stateOverflow};
  }
  if (ast.result.kind == ReductionKind::SosTdf2) {
    if (ast.results.size() != 2 || !ast.primaryResult().tensor || !ast.results[1].tensor ||
        ast.primaryResult().type != SourceType::F32 || !lhsParameter || !rhsParameter ||
        !thirdParameter || !fourthParameter ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isTensor(); })) {
      diagnostics.error(ast.result.position,
                        "sos_tdf2 requires four f32 tensor parameters and two f32 tensor results");
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 2) ||
        !hasRank(thirdParameter->shape, 1) || !hasRank(fourthParameter->shape, 2) ||
        !hasRank(ast.primaryResult().shape, 1) || !hasRank(ast.results[1].shape, 2)) {
      diagnostics.error(ast.result.position, "sos_tdf2 requires input/output rank 1, "
                                             "coefficients/state rank 2, and scales rank 1");
      return std::nullopt;
    }
    const std::optional<int64_t> &inputExtent = lhsParameter->shape[0];
    const std::optional<int64_t> &outputExtent = ast.primaryResult().shape[0];
    if (inputExtent.has_value() != outputExtent.has_value() ||
        (inputExtent && *inputExtent != *outputExtent)) {
      diagnostics.error(ast.result.position, "sos_tdf2 input and output chunk extents must match");
      return std::nullopt;
    }
    const std::optional<int64_t> &sections = rhsParameter->shape[0];
    if (!sections || *sections < 1) {
      diagnostics.error(ast.result.position,
                        "sos_tdf2 requires a static coefficient section count of at least one");
      return std::nullopt;
    }
    if (rhsParameter->shape[1] != std::optional<int64_t>(5) ||
        thirdParameter->shape[0] != sections || fourthParameter->shape[0] != sections ||
        fourthParameter->shape[1] != std::optional<int64_t>(2) ||
        ast.results[1].shape[0] != sections ||
        ast.results[1].shape[1] != std::optional<int64_t>(2)) {
      diagnostics.error(ast.result.position,
                        "sos_tdf2 requires coefficients [S,5], scales [S], and state [S,2]");
      return std::nullopt;
    }
    auto contract = parseFpContract(ast.result.fpContract);
    if (!contract) {
      diagnostics.error(ast.result.position, llvm::Twine("unsupported floating-point contract '") +
                                                 ast.result.fpContract + "'");
      return std::nullopt;
    }
    // The biquad event graph has no realization gate for `fast`, so the
    // operation contract admits only the two exact modes.
    if (*contract == ondsp::FpContractMode::Fast) {
      diagnostics.error(ast.result.position, "sos_tdf2 admits only contract=off or contract=fma");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
  }
  if (ast.result.kind == ReductionKind::Matmul) {
    bool isFloat = ast.primaryResult().type == SourceType::F32;
    bool isQ31 = ast.primaryResult().type == SourceType::Q31;
    if (ast.results.size() != 1 || !ast.primaryResult().tensor ||
        (ast.primaryResult().type != SourceType::Q15 && !isFloat && !isQ31) || !lhsParameter ||
        !rhsParameter || constexprCount != 0 ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isTensor(); })) {
      diagnostics.error(ast.result.position,
                        "matmul requires two Q15, Q31 or f32 tensor parameters and a matching "
                        "tensor result");
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 2) || !hasRank(rhsParameter->shape, 2) ||
        !hasRank(ast.primaryResult().shape, 2)) {
      diagnostics.error(ast.result.position, "matmul requires rank-2 tensors");
      return std::nullopt;
    }
    if (llvm::any_of(lhsParameter->shape,
                     [](std::optional<int64_t> extent) { return !extent.has_value(); }) ||
        llvm::any_of(rhsParameter->shape,
                     [](std::optional<int64_t> extent) { return !extent.has_value(); }) ||
        llvm::any_of(ast.primaryResult().shape,
                     [](std::optional<int64_t> extent) { return !extent.has_value(); })) {
      diagnostics.error(ast.result.position, "matmul currently requires static extents");
      return std::nullopt;
    }
    int64_t rows = *lhsParameter->shape[0];
    int64_t inner = *lhsParameter->shape[1];
    int64_t columns = *rhsParameter->shape[1];
    if (*rhsParameter->shape[0] != inner) {
      diagnostics.error(ast.result.position,
                        "matmul inner extents must match: lhs columns and rhs rows");
      return std::nullopt;
    }
    auto inRange = [](int64_t extent) { return extent >= 1 && extent <= 64; };
    if (!inRange(rows) || !inRange(inner) || !inRange(columns)) {
      diagnostics.error(ast.result.position, "matmul currently requires all extents in [1, 64]");
      return std::nullopt;
    }
    if (*ast.primaryResult().shape[0] != rows || *ast.primaryResult().shape[1] != columns) {
      diagnostics.error(ast.result.position, "matmul result shape must be lhs rows by rhs columns");
      return std::nullopt;
    }
    if (isFloat) {
      auto contract = parseFpContract(ast.result.fpContract);
      if (!contract) {
        diagnostics.error(ast.result.position,
                          llvm::Twine("unsupported floating-point contract '") +
                              ast.result.fpContract + "'");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
    }
    ondsp::RoundingMode rounding = ondsp::RoundingMode::NearestEven;
    if (!ast.result.rounding.empty()) {
      std::optional<ondsp::RoundingMode> parsed = parseRounding(ast.result.rounding);
      if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven &&
                      *parsed != ondsp::RoundingMode::TowardNegative &&
                      *parsed != ondsp::RoundingMode::NearestTiesPositive)) {
        diagnostics.error(ast.result.position,
                          "matmul rounding must be nearest_even, toward_negative, or "
                          "nearest_ties_positive");
        return std::nullopt;
      }
      rounding = *parsed;
    }
    // The raw high half is the Q31 target's product: one floor per term, no
    // second rounding, the sum read out by a doubling.
    bool rawHigh = ast.result.product == "raw_high";
    if (!ast.result.product.empty() && ast.result.product != "full" && !rawHigh) {
      diagnostics.error(ast.result.position, llvm::Twine("unsupported product selection '") +
                                                 ast.result.product + "'; use full or raw_high");
      return std::nullopt;
    }
    if (rawHigh && !isQ31) {
      diagnostics.error(ast.result.position,
                        "product=raw_high is the Q31 matmul profile: each term is the raw high "
                        "half of the product, accumulated at frac 30");
      return std::nullopt;
    }
    if (rawHigh && !ast.result.inputRounding.empty()) {
      diagnostics.error(ast.result.position,
                        "a raw-high matmul has no product rounding to declare: the high half is "
                        "a floor");
      return std::nullopt;
    }
    // The product boundary exists only where an exact K-sum would not fit i64,
    // so the binding declares its rounding exactly there and refuses it
    // elsewhere. K is the inner extent, already validated as static above.
    std::optional<ondsp::RoundingMode> productRounding;
    bool hasProductBoundary =
        !rawHigh && ir::getReductionProductShift(isQ31 ? 32 : 16, *lhsParameter->shape[1]) > 0;
    if (!ast.result.inputRounding.empty()) {
      if (!hasProductBoundary) {
        diagnostics.error(ast.result.position,
                          "matmul at this width and inner extent has no product boundary to round");
        return std::nullopt;
      }
      std::optional<ondsp::RoundingMode> parsed = parseRounding(ast.result.inputRounding);
      if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven &&
                      *parsed != ondsp::RoundingMode::TowardNegative &&
                      *parsed != ondsp::RoundingMode::NearestTiesPositive)) {
        diagnostics.error(ast.result.position,
                          "matmul product_rounding must be nearest_even, toward_negative, or "
                          "nearest_ties_positive");
        return std::nullopt;
      }
      productRounding = *parsed;
    } else if (hasProductBoundary) {
      productRounding = ondsp::RoundingMode::NearestEven;
    }
    CheckedKernel checked{std::move(ast), std::nullopt, rounding, std::nullopt, std::nullopt};
    checked.productRounding = productRounding;
    return checked;
  }
  if (ast.result.kind == ReductionKind::Lms) {
    bool isFloat = ast.primaryResult().type == SourceType::F32;
    bool isQ31 = ast.primaryResult().type == SourceType::Q31;
    if (ast.results.size() != 2 || !ast.primaryResult().tensor || !ast.results[1].tensor ||
        (ast.primaryResult().type != SourceType::Q15 && !isFloat && !isQ31) || !lhsParameter ||
        !rhsParameter || !thirdParameter || constexprCount != 0 ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isTensor(); })) {
      diagnostics.error(ast.result.position,
                        "lms requires three Q15, Q31 or f32 tensor parameters and two matching "
                        "tensor results");
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 1) ||
        !hasRank(thirdParameter->shape, 1) || !hasRank(ast.primaryResult().shape, 1) ||
        !hasRank(ast.results[1].shape, 1)) {
      diagnostics.error(ast.result.position, "lms currently requires rank-1 tensors");
      return std::nullopt;
    }
    const std::optional<int64_t> &inputExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &desiredExtent = getRankOneExtent(rhsParameter->shape);
    const std::optional<int64_t> &weightExtent = getRankOneExtent(thirdParameter->shape);
    const std::optional<int64_t> &errorExtent = getRankOneExtent(ast.primaryResult().shape);
    const std::optional<int64_t> &adaptedExtent = getRankOneExtent(ast.results[1].shape);
    if (!inputExtent || !desiredExtent || !weightExtent || !errorExtent || !adaptedExtent) {
      diagnostics.error(ast.result.position, "lms currently requires static extents");
      return std::nullopt;
    }
    if (*inputExtent < 1 || *inputExtent > 4096) {
      diagnostics.error(ast.result.position, "lms currently requires a sample extent in [1, 4096]");
      return std::nullopt;
    }
    if (*desiredExtent != *inputExtent || *errorExtent != *inputExtent) {
      diagnostics.error(ast.result.position, "lms input, desired, and error extents must match");
      return std::nullopt;
    }
    if (*weightExtent < 1 || *weightExtent > 64) {
      diagnostics.error(ast.result.position, "lms currently requires a weight extent in [1, 64]");
      return std::nullopt;
    }
    if (*adaptedExtent != *weightExtent) {
      diagnostics.error(ast.result.position,
                        "lms adapted weights must have the initial weight extent");
      return std::nullopt;
    }
    if (ast.result.normalized && (isFloat || isQ31)) {
      diagnostics.error(ast.result.position, "nlms is the Q15 profile for now");
      return std::nullopt;
    }
    if (ast.result.normalized && (ast.result.epsilon < 1 || ast.result.epsilon > 32767)) {
      diagnostics.error(ast.result.position,
                        "nlms epsilon must be a raw Q1.15 value in [1, 32767]");
      return std::nullopt;
    }
    if (isFloat) {
      auto [constant, refusal] = roundRationalToF32(ast.result.stepSize, ast.result.fpConstantDen);
      if (refusal != RationalRefusal::None) {
        diagnostics.error(ast.result.position,
                          llvm::Twine("f32 lms step size: ") + describeRationalRefusal(refusal));
        return std::nullopt;
      }
      // The fixed profile admits only the non-negative raw Q1.15 range; the
      // f32 profile matches it rather than widening by omission.
      if (constant < 0.0f) {
        diagnostics.error(ast.result.position, "f32 lms step size must not be negative");
        return std::nullopt;
      }
      ast.result.fpConstant = constant;
      auto contract = parseFpContract(ast.result.fpContract);
      if (!contract) {
        diagnostics.error(ast.result.position,
                          llvm::Twine("unsupported floating-point contract '") +
                              ast.result.fpContract + "'");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
    }
    unsigned storageWidth = isQ31 ? 32u : 16u;
    int64_t stepCeiling = (int64_t(1) << (storageWidth - 1)) - 1;
    if (ast.result.stepSize < 0 || ast.result.stepSize > stepCeiling) {
      diagnostics.error(ast.result.position, llvm::Twine("lms step size must be a raw signed Q1.") +
                                                 llvm::Twine(storageWidth - 1) + " value in [0, " +
                                                 llvm::Twine(stepCeiling) + "]");
      return std::nullopt;
    }
    // Only the tap sum can leave i64, so the product boundary and its declared
    // rounding exist exactly where that sum would.
    std::optional<ondsp::RoundingMode> productRounding;
    bool hasProductBoundary =
        ir::getReductionProductShift(storageWidth, *thirdParameter->shape[0]) > 0;
    if (!ast.result.inputRounding.empty()) {
      if (!hasProductBoundary) {
        diagnostics.error(ast.result.position,
                          "lms at this width and tap count has no product boundary to round");
        return std::nullopt;
      }
      std::optional<ondsp::RoundingMode> parsed = parseRounding(ast.result.inputRounding);
      if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven &&
                      *parsed != ondsp::RoundingMode::TowardNegative)) {
        diagnostics.error(ast.result.position,
                          "lms product_rounding must be nearest_even or toward_negative");
        return std::nullopt;
      }
      productRounding = *parsed;
    } else if (hasProductBoundary) {
      productRounding = ondsp::RoundingMode::NearestEven;
    }
    CheckedKernel checked{std::move(ast), std::nullopt, ondsp::RoundingMode::NearestEven,
                          std::nullopt, std::nullopt};
    checked.productRounding = productRounding;
    return checked;
  }
  if (llvm::any_of(ast.parameters,
                   [](const ParameterAst &parameter) {
                     return (parameter.isBuffer() || parameter.isTensor()) &&
                            !hasRank(parameter.shape, 1);
                   }) ||
      (ast.primaryResult().tensor && !hasRank(ast.primaryResult().shape, 1)) ||
      (ast.results.size() > 1 && ast.results[1].tensor && !hasRank(ast.results[1].shape, 1))) {
    diagnostics.error(ast.result.position,
                      "this builtin currently requires rank-1 shaped parameters and results");
    return std::nullopt;
  }
  if (ast.results.size() != 1) {
    diagnostics.error(ast.result.position,
                      "multiple results are currently supported only by butterfly and stateful "
                      "DSP builtins");
    return std::nullopt;
  }
  if (isComposableKernel(ast.result.kind, ast.primaryResult().type)) {
    struct ComposedType {
      SourceType elementType;
      int64_t extent;
    };
    llvm::StringMap<unsigned> parameterUses;
    llvm::StringMap<ComposedType> localTypes;
    llvm::StringMap<APInt> localConstants;
    std::optional<SourceType> literalHint;
    std::function<std::optional<ComposedType>(ExpressionAst &)> checkComposedExpression;
    std::function<std::optional<ComposedType>(BuiltinCallAst &)> checkComposedCall;
    checkComposedExpression = [&](ExpressionAst &expression) -> std::optional<ComposedType> {
      if (expression.isParameterReference()) {
        auto local = localTypes.find(expression.parameter);
        if (local != localTypes.end())
          return local->second;
        auto parameter = parametersByName.find(expression.parameter);
        if (parameter == parametersByName.end()) {
          diagnostics.error(expression.position,
                            llvm::Twine("unknown operand '") + expression.parameter + "'");
          return std::nullopt;
        }
        ++parameterUses[expression.parameter];
        const ParameterAst *value = parameter->second;
        if (value->isScalar() &&
            (value->type == SourceType::I32 || value->type == SourceType::I64 ||
             value->type == SourceType::Bool || value->type == SourceType::F32))
          return ComposedType{value->type, 0};
        if (!value->isTensor() || !hasRank(value->shape, 1)) {
          diagnostics.error(expression.position,
                            "composable builtins currently require rank-1 tensor operands");
          return std::nullopt;
        }
        const std::optional<int64_t> &extent = getRankOneExtent(value->shape);
        if (!extent) {
          diagnostics.error(expression.position,
                            "composable builtins currently require static operand extents");
          return std::nullopt;
        }
        return ComposedType{value->type, *extent};
      }

      std::optional<ComposedType> nested = checkComposedCall(*expression.call);
      // Decoded bits carry no numeric reading any builtin consumes.
      if (nested && nested->elementType == SourceType::U8) {
        diagnostics.error(expression.position, "u8 decoded bits can only be returned");
        return std::nullopt;
      }
      return nested;
    };
    std::function<bool(const ExpressionAst &)> needsLiteralContext;
    needsLiteralContext = [&](const ExpressionAst &expression) {
      if (expression.isParameterReference())
        return false;
      const auto &call = *expression.call;
      if (call.kind != ReductionKind::Primitive)
        return getOrdinaryArithmetic(call.kind) && llvm::all_of(call.operands, needsLiteralContext);
      if (call.primitive == PrimitiveKind::Literal)
        return call.spelling != "True" && call.spelling != "False" &&
               call.spelling.find_first_of(".eE") == std::string::npos;
      if (call.primitive != PrimitiveKind::Identity && call.primitive != PrimitiveKind::Pos &&
          !getOperatorBuiltin(call.primitive))
        return false;
      return llvm::all_of(call.operands, needsLiteralContext);
    };
    auto constant = [&](const ExpressionAst &expression) -> std::optional<APInt> {
      if (expression.call)
        return expression.call->integerConstant;
      auto found = localConstants.find(expression.parameter);
      return found == localConstants.end() ? std::nullopt : std::optional<APInt>(found->second);
    };
    // Induction variables whose range proves `0 <= v < len(parameter)`.
    llvm::StringMap<std::string> provenIndices;
    auto sameType = [](ComposedType lhs, ComposedType rhs) {
      return lhs.elementType == rhs.elementType && lhs.extent == rhs.extent;
    };
    std::function<void(const BuiltinCallAst &, size_t, SmallVector<std::string> &)>
        collectAssigned =
            [&](const BuiltinCallAst &owner, size_t begin, SmallVector<std::string> &names) {
              for (size_t i = begin; i < owner.bindingNames.size(); ++i) {
                if (owner.bindingNames[i].empty())
                  collectAssigned(*owner.operands[i].call, 2, names);
                else if (!llvm::is_contained(names, owner.bindingNames[i]))
                  names.push_back(owner.bindingNames[i]);
              }
            };
    std::function<bool(BuiltinCallAst &)> checkLoop;
    auto checkStatements = [&](BuiltinCallAst &owner, size_t begin) {
      for (size_t i = begin; i < owner.bindingNames.size(); ++i) {
        ExpressionAst &statement = owner.operands[i];
        const std::string &name = owner.bindingNames[i];
        if (name.empty()) {
          if (!checkLoop(*statement.call))
            return false;
          continue;
        }
        std::optional<ComposedType> type = checkComposedExpression(statement);
        if (!type)
          return false;
        auto previous = localTypes.find(name);
        if (previous != localTypes.end() && !sameType(previous->second, *type)) {
          diagnostics.error(statement.position, "a reassignment must keep the local's type");
          return false;
        }
        localTypes[name] = *type;
        if (auto value = constant(statement))
          localConstants.insert_or_assign(name, *value);
        else
          localConstants.erase(name);
      }
      return true;
    };
    checkLoop = [&](BuiltinCallAst &loop) {
      SmallVector<std::optional<ComposedType>, 2> bounds(2);
      std::optional<SourceType> domain;
      for (size_t i = 0; i < 2; ++i) {
        if (needsLiteralContext(loop.operands[i]))
          continue;
        bounds[i] = checkComposedExpression(loop.operands[i]);
        if (!bounds[i])
          return false;
        domain = domain.value_or(bounds[i]->elementType);
      }
      auto savedHint = literalHint;
      literalHint = domain.value_or(SourceType::I32);
      for (size_t i = 0; i < 2; ++i)
        if (!bounds[i] && !(bounds[i] = checkComposedExpression(loop.operands[i])))
          return false;
      literalHint = savedHint;
      SourceType type = bounds[0]->elementType;
      if (!sameType(*bounds[0], *bounds[1]) || bounds[0]->extent ||
          (type != SourceType::Index && type != SourceType::I32 && type != SourceType::I64)) {
        diagnostics.error(loop.position, "range bounds must share one index, i32 or i64 type");
        return false;
      }
      if (!llvm::isIntN(type == SourceType::I64 ? 64 : 32, loop.step)) {
        diagnostics.error(loop.position, "range step is out of range for the bound type");
        return false;
      }

      SmallVector<std::string> assigned;
      collectAssigned(loop, 2, assigned);
      loop.carriedNames.clear();
      for (const std::string &name : assigned) {
        auto outer = localTypes.find(name);
        if (outer == localTypes.end())
          continue;
        if (outer->second.extent) {
          diagnostics.error(loop.position, "a loop can only carry scalar locals");
          return false;
        }
        loop.carriedNames.push_back(name);
        localConstants.erase(name);
      }
      auto outerTypes = localTypes;
      auto outerConstants = localConstants;
      const std::string &induction = loop.bindingNames[0];
      localTypes[induction] = ComposedType{type, 0};
      const BuiltinCallAst *stop = loop.operands[1].call.get();
      std::optional<APInt> start = constant(loop.operands[0]);
      if (loop.step > 0 && start && !start->isNegative() && stop &&
          stop->kind == ReductionKind::Primitive && stop->primitive == PrimitiveKind::Length)
        provenIndices[induction] = stop->operands.front().parameter;
      bool checked = checkStatements(loop, 2);
      provenIndices.erase(induction);
      localTypes = std::move(outerTypes);
      localConstants = std::move(outerConstants);
      return checked;
    };
    // The parameter `len` and subscripts read; complex_f32 is excluded because
    // its storage extent is not its element count.
    auto getIndexedParameter = [&](const ExpressionAst &base) -> const ParameterAst * {
      auto parameter = parametersByName.find(base.parameter);
      if (parameter == parametersByName.end() || !hasRank(parameter->second->shape, 1) ||
          parameter->second->type != SourceType::F32) {
        diagnostics.error(base.position, "len and indexing currently require a rank-one f32 "
                                         "buffer or tensor parameter");
        return nullptr;
      }
      ++parameterUses[base.parameter];
      return parameter->second;
    };

    auto checkPrimitive = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      auto reject = [&](const Twine &message) -> std::optional<ComposedType> {
        diagnostics.error(call.position, message);
        return std::nullopt;
      };
      auto finish = [&](ComposedType type) -> std::optional<ComposedType> {
        call.expressionType = type.elementType;
        call.expressionExtent = type.extent;
        return type;
      };
      auto op = call.primitive;
      if (op == PrimitiveKind::Scope) {
        auto saved = localTypes;
        auto savedConstants = localConstants;
        std::optional<ComposedType> result;
        if (checkStatements(call, 0)) {
          result = checkComposedExpression(call.operands.back());
          call.integerConstant = constant(call.operands.back());
        }
        localTypes = std::move(saved);
        localConstants = std::move(savedConstants);
        return result ? finish(*result) : std::nullopt;
      }
      if (op == PrimitiveKind::Length) {
        if (!getIndexedParameter(call.operands.front()))
          return std::nullopt;
        return finish({SourceType::Index, 0});
      }
      if (op == PrimitiveKind::Element) {
        const ParameterAst *parameter = getIndexedParameter(call.operands.front());
        if (!parameter)
          return std::nullopt;
        auto savedHint = literalHint;
        literalHint = SourceType::Index;
        std::optional<ComposedType> index = checkComposedExpression(call.operands[1]);
        literalHint = savedHint;
        if (!index)
          return std::nullopt;
        if (index->extent ||
            (index->elementType != SourceType::Index && index->elementType != SourceType::I32 &&
             index->elementType != SourceType::I64))
          return reject("an index must be an index, i32 or i64 scalar");
        const ExpressionAst &position = call.operands[1];
        call.inBounds = position.isParameterReference() &&
                        provenIndices.lookup(position.parameter) == parameter->name;
        return finish({SourceType::F32, 0});
      }
      if (op == PrimitiveKind::Literal) {
        SourceType type = literalHint.value_or(ast.primaryResult().type);
        if (call.spelling == "True" || call.spelling == "False") {
          call.integerConstant = APInt(1, call.spelling == "True");
          return finish({SourceType::Bool, 0});
        }
        bool floating = call.spelling.find_first_of(".eE") != std::string::npos;
        if (floating)
          type = SourceType::F32;
        if (type == SourceType::Bool)
          type = SourceType::I32;
        if (type == SourceType::F32) {
          llvm::APFloat value(llvm::APFloat::IEEEsingle());
          auto status = value.convertFromString(call.spelling, llvm::APFloat::rmNearestTiesToEven);
          if (!status) {
            llvm::consumeError(status.takeError());
            return reject("invalid f32 literal");
          }
          if (*status & llvm::APFloat::opOverflow)
            return reject("f32 literal is out of range");
          call.fpConstant = value.convertToFloat();
          return finish({type, 0});
        }
        if (type != SourceType::I32 && type != SourceType::I64 && type != SourceType::Index)
          return reject("ordinary literal requires an integer or f32 context");
        // An index literal must mean the same value at every target index width.
        unsigned width = type == SourceType::I64 ? 64 : 32;
        StringRef spelling(call.spelling);
        bool negative = spelling.consume_front("-");
        APInt magnitude(128, 0);
        unsigned radix = spelling.consume_front("0o") ? 8 : 10;
        if (spelling.getAsInteger(radix, magnitude) || magnitude.getActiveBits() > width)
          return reject("integer literal is out of range");
        magnitude = magnitude.zextOrTrunc(128);
        APInt limit = APInt::getSignedMaxValue(width).zext(128);
        if (negative)
          ++limit;
        if (magnitude.ugt(limit))
          return reject("integer literal is out of range");
        APInt value = negative ? -magnitude.trunc(width) : magnitude.trunc(width);
        call.integerConstant = type == SourceType::Index ? value.sext(64) : value;
        return finish({type, 0});
      }
      if (op == PrimitiveKind::Typed) {
        auto savedHint = literalHint;
        literalHint = call.target;
        auto actual = checkComposedExpression(call.operands.front());
        literalHint = savedHint;
        if (!actual)
          return std::nullopt;
        if (actual->elementType != call.target || actual->extent != call.declaredExtent)
          return reject("call argument does not match the declared parameter type");
        call.integerConstant = constant(call.operands.front());
        return finish(*actual);
      }
      SmallVector<std::optional<ComposedType>, 2> operandTypes(call.operands.size());
      std::optional<SourceType> domain;
      for (auto [index, operand] : llvm::enumerate(call.operands)) {
        if (needsLiteralContext(operand))
          continue;
        operandTypes[index] = checkComposedExpression(operand);
        if (!operandTypes[index])
          return std::nullopt;
        domain = operandTypes[index]->elementType;
        break;
      }
      std::optional<ReductionKind> fixedKind = getOperatorBuiltin(op);
      if (domain && (*domain == SourceType::Q15 || *domain == SourceType::Q31) && fixedKind) {
        if (op == PrimitiveKind::Neg) {
          call.kind = *fixedKind;
        } else {
          SmallVector<Operand, 2> operands;
          for (ExpressionAst &expression : call.operands) {
            bool literal = expression.call && expression.call->kind == ReductionKind::Primitive &&
                           expression.call->primitive == PrimitiveKind::Literal;
            if (!literal) {
              operands.push_back({std::move(expression), 0, call.position});
              continue;
            }
            StringRef spelling(expression.call->spelling);
            bool negative = spelling.consume_front("-");
            uint64_t magnitude = 0;
            if (getIntegerTokenValue(spelling, magnitude) || magnitude > uint64_t(INT64_MAX))
              return reject("fixed tensor arithmetic requires a raw integer literal");
            operands.push_back({std::nullopt, negative ? -int64_t(magnitude) : int64_t(magnitude),
                                expression.position});
          }
          auto fixed = buildFixedBinary(*fixedKind, std::move(operands[0]), std::move(operands[1]),
                                        call.position, diagnostics);
          if (!fixed)
            return std::nullopt;
          call = std::move(*fixed);
        }
        return checkComposedCall(call);
      }
      auto savedHint = literalHint;
      if (domain)
        literalHint = domain;
      auto lhs = operandTypes[0] ? operandTypes[0] : checkComposedExpression(call.operands.front());
      if (!lhs) {
        literalHint = savedHint;
        return std::nullopt;
      }
      bool logical =
          op == PrimitiveKind::And || op == PrimitiveKind::Or || op == PrimitiveKind::Not;
      bool comparison = isComparison(op);
      std::optional<ComposedType> rhs;
      if (call.operands.size() == 2)
        rhs = operandTypes[1] ? operandTypes[1] : checkComposedExpression(call.operands[1]);
      literalHint = savedHint;
      if (call.operands.size() == 2 && !rhs)
        return std::nullopt;
      if (op == PrimitiveKind::Identity) {
        call.integerConstant = constant(call.operands.front());
        return finish(*lhs);
      }
      if (logical) {
        if (lhs->elementType != SourceType::Bool || lhs->extent ||
            (rhs && (rhs->elementType != SourceType::Bool || rhs->extent)))
          return reject("logical operators require scalar bool operands");
        return finish({SourceType::Bool, 0});
      }
      bool floating = lhs->elementType == SourceType::F32;
      bool index = lhs->elementType == SourceType::Index;
      bool integer =
          index || lhs->elementType == SourceType::I32 || lhs->elementType == SourceType::I64;
      if (!floating && !integer && !(comparison && lhs->elementType == SourceType::Bool))
        return reject("ordinary arithmetic requires i32, i64 or f32 operands");
      int64_t extent = lhs->extent;
      if (rhs) {
        if (rhs->elementType != lhs->elementType)
          return reject(
              "arithmetic operands must have the same element type; no implicit conversion");
        if (rhs->extent && lhs->extent && rhs->extent != lhs->extent)
          return reject("arithmetic tensor operands must have matching extents");
        extent = std::max(extent, rhs->extent);
      }
      if (extent > 4096)
        return reject("f32 arithmetic requires static rank-1 extents in [1, 4096]");
      if (comparison) {
        if (extent)
          return reject("comparisons require scalar operands");
        if (lhs->elementType == SourceType::Bool && op != PrimitiveKind::Eq &&
            op != PrimitiveKind::Ne)
          return reject("bool only supports equality and inequality comparisons");
        return finish({SourceType::Bool, 0});
      }
      if (index && (op == PrimitiveKind::Mul || op == PrimitiveKind::Div))
        return reject("index arithmetic supports only + and -");
      auto a = constant(call.operands.front());
      auto b = rhs ? constant(call.operands[1]) : std::nullopt;
      if (integer && op == PrimitiveKind::Div) {
        if (b && b->isZero())
          return reject("integer division by zero");
        if (a && b && a->isMinSignedValue() && b->isAllOnes())
          return reject("integer division overflows: MIN / -1");
      }
      if (integer && a && (!rhs || b)) {
        switch (op) {
        case PrimitiveKind::Pos:
          call.integerConstant = *a;
          break;
        case PrimitiveKind::Neg:
          call.integerConstant = -*a;
          break;
        case PrimitiveKind::Add:
          call.integerConstant = *a + *b;
          break;
        case PrimitiveKind::Sub:
          call.integerConstant = *a - *b;
          break;
        case PrimitiveKind::Mul:
          call.integerConstant = *a * *b;
          break;
        case PrimitiveKind::Div:
          call.integerConstant = a->sdiv(*b);
          break;
        default:
          break;
        }
        if (index && call.integerConstant && !call.integerConstant->isSignedIntN(32))
          return reject("constant index expression leaves the signed 32-bit range");
      }
      return finish({lhs->elementType, extent});
    };

    // A fir_filter stage may feed the FFT chain: static Q15 or Q31 tensors,
    // the valid boundary, and the executable export profile of that width.
    // Its coefficients are a tensor parameter or a design builtin.
    auto checkNestedFirFilter = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      if (call.operands.size() != 2) {
        diagnostics.error(call.position, "builtin operand count does not match its contract");
        return std::nullopt;
      }
      const ExpressionAst &inputOperand = call.operands[0];
      if (!inputOperand.isParameterReference()) {
        diagnostics.error(inputOperand.position,
                          "a composed fir_filter currently takes its input from a parameter");
        return std::nullopt;
      }
      auto inputEntry = parametersByName.find(inputOperand.parameter);
      if (inputEntry == parametersByName.end()) {
        diagnostics.error(inputOperand.position,
                          llvm::Twine("unknown builtin operand '") + inputOperand.parameter + "'");
        return std::nullopt;
      }
      ++parameterUses[inputOperand.parameter];
      const ParameterAst *input = inputEntry->second;
      bool isQ31 = input->type == SourceType::Q31;
      if (!input->isTensor() || (input->type != SourceType::Q15 && !isQ31) ||
          !hasRank(input->shape, 1) || !getRankOneExtent(input->shape)) {
        diagnostics.error(inputOperand.position,
                          "a composed fir_filter currently requires a static rank-1 Q15 or Q31 "
                          "tensor input");
        return std::nullopt;
      }
      int64_t inputExtent = *getRankOneExtent(input->shape);
      int64_t tapCount = 0;
      const ExpressionAst &coefficientOperand = call.operands[1];
      if (coefficientOperand.isParameterReference()) {
        auto entry = parametersByName.find(coefficientOperand.parameter);
        if (entry == parametersByName.end()) {
          diagnostics.error(coefficientOperand.position,
                            llvm::Twine("unknown reduction operand '") +
                                coefficientOperand.parameter + "'");
          return std::nullopt;
        }
        ++parameterUses[coefficientOperand.parameter];
        const ParameterAst *coefficients = entry->second;
        if (!coefficients->isTensor() || coefficients->type != input->type ||
            !hasRank(coefficients->shape, 1) || !getRankOneExtent(coefficients->shape)) {
          diagnostics.error(coefficientOperand.position,
                            "composed fir_filter coefficients currently require a static rank-1 "
                            "tensor parameter of the input's width or a coefficient design");
          return std::nullopt;
        }
        tapCount = *getRankOneExtent(coefficients->shape);
      } else if (coefficientOperand.call->kind == ReductionKind::Lowpass) {
        const BuiltinCallAst &design = *coefficientOperand.call;
        if (design.taps < 3 || design.taps > 4095 || design.taps % 2 == 0) {
          diagnostics.error(design.position,
                            "lowpass currently requires an odd tap count in [3, 4095]");
          return std::nullopt;
        }
        // 0 < num/den < 1/2 strictly, compared without overflow.
        if (design.cutoffNum < 1 || design.cutoffDen < 1 ||
            design.cutoffNum > (design.cutoffDen - 1) / 2) {
          diagnostics.error(design.position,
                            "lowpass cutoff must satisfy 0 < num/den < 1/2 strictly");
          return std::nullopt;
        }
        tapCount = design.taps;
      } else if (isWindowDesignKind(coefficientOperand.call->kind)) {
        const BuiltinCallAst &design = *coefficientOperand.call;
        if (design.taps < 2 || design.taps > 4096) {
          diagnostics.error(design.position, llvm::Twine(describeDesignKind(design.kind)) +
                                                 " requires a tap count in [2, 4096]");
          return std::nullopt;
        }
        // 0 < num/den <= 50, compared without overflow: the sign test runs
        // first so `50 * den` is only formed on a proven-positive denominator.
        if (design.kind == ReductionKind::Kaiser &&
            (design.betaNum < 1 || design.betaDen < 1 ||
             (design.betaDen <= std::numeric_limits<int64_t>::max() / 50 &&
              design.betaNum > 50 * design.betaDen))) {
          diagnostics.error(design.position, "kaiser beta must be a positive rational in (0, 50]");
          return std::nullopt;
        }
        tapCount = design.taps;
      } else {
        diagnostics.error(coefficientOperand.position,
                          "composed fir_filter coefficients currently require a static rank-1 "
                          "tensor parameter of the input's width or a coefficient design");
        return std::nullopt;
      }
      if (call.boundary != "valid") {
        diagnostics.error(call.position, "a composed fir_filter currently supports boundary=valid");
        return std::nullopt;
      }
      if (inputExtent < tapCount) {
        diagnostics.error(call.position,
                          "valid-boundary fir_filter requires input extent >= tap count");
        return std::nullopt;
      }
      // A composed stage takes the same default an uncomposed one takes;
      // the tap count is already static here, so the inference is identical,
      // and a Q31 stage spells its contract for the reason an uncomposed one does.
      if (call.accumulatorAuto) {
        if (isQ31) {
          diagnostics.error(call.position, "a Q31 fir_filter stage requires an explicit "
                                           "accumulator, rounding, and overflow policy");
          return std::nullopt;
        }
        call.accumulatorAuto = false;
        call.accumulatorWidth = inferQ15FullAccumulatorWidth(static_cast<uint64_t>(tapCount));
      } else if (call.accumulatorWidth != (isQ31 ? 64u : 40u)) {
        diagnostics.error(call.position,
                          isQ31 ? "the executable Q31 profile requires exact accumulator width 64"
                                : "the executable Q15 profile requires exact accumulator width 40");
        return std::nullopt;
      }
      if (!parseOverflow(call.updateOverflow)) {
        diagnostics.error(call.position, llvm::Twine("unsupported update overflow mode '") +
                                             call.updateOverflow + "'");
        return std::nullopt;
      }
      std::optional<ondsp::RoundingMode> rounding = parseRounding(call.rounding);
      if (!rounding) {
        diagnostics.error(call.position,
                          llvm::Twine("unsupported rounding mode '") + call.rounding + "'");
        return std::nullopt;
      }
      if (!isDeclaredExportRounding(*rounding)) {
        diagnostics.error(
            call.position,
            "export rounding must be nearest_even, nearest_ties_positive, toward_negative, or "
            "toward_zero");
        return std::nullopt;
      }
      if (!parseOverflow(call.destinationOverflow)) {
        diagnostics.error(call.position, llvm::Twine("unsupported destination overflow mode '") +
                                             call.destinationOverflow + "'");
        return std::nullopt;
      }
      return ComposedType{input->type, inputExtent - tapCount + 1};
    };
    // The elementwise family: one uniform-Q width in, the same width out,
    // extents equal. Only the two boundary attributes vary between members,
    // so one checker covers all seven and a new member cannot forget a rule.
    // The two attribute ranges follow the width because they are raw values
    // in the declared format, not abstract quantities.
    // The language defaults for a declared boundary, then the mode names.
    auto checkBoundaryPolicies = [&](BuiltinCallAst &call) -> bool {
      if (call.rounding.empty())
        call.rounding = "nearest_ties_positive";
      if (call.destinationOverflow.empty())
        call.destinationOverflow = "saturate";
      if (!parseRounding(call.rounding)) {
        diagnostics.error(call.position,
                          llvm::Twine("unsupported rounding mode '") + call.rounding + "'");
        return false;
      }
      if (!parseOverflow(call.destinationOverflow)) {
        diagnostics.error(call.position, llvm::Twine("unsupported overflow mode '") +
                                             call.destinationOverflow + "'");
        return false;
      }
      return true;
    };
    auto checkElementwise = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      std::optional<ComposedType> lhs = checkComposedExpression(call.operands.front());
      if (!lhs)
        return std::nullopt;
      auto primitive = getOrdinaryArithmetic(call.kind);
      if (primitive &&
          (lhs->elementType == SourceType::F32 || lhs->elementType == SourceType::I32 ||
           lhs->elementType == SourceType::I64)) {
        if (!call.rounding.empty() || !call.destinationOverflow.empty()) {
          diagnostics.error(call.position,
                            "ordinary arithmetic does not accept fixed-point policies");
          return std::nullopt;
        }
        call.primitive = *primitive;
        call.kind = ReductionKind::Primitive;
        return checkPrimitive(call);
      }
      if (lhs->elementType != SourceType::Q15 && lhs->elementType != SourceType::Q31) {
        diagnostics.error(call.position, "elementwise builtins require q15 or q31 operand "
                                         "elements");
        return std::nullopt;
      }
      if (lhs->extent < 1 || lhs->extent > 4096) {
        diagnostics.error(call.position,
                          "elementwise builtins currently require an extent in [1, 4096]");
        return std::nullopt;
      }
      int64_t fractionalBits = lhs->elementType == SourceType::Q31 ? 31 : 15;
      if (isBinaryElementwiseKind(call.kind)) {
        std::optional<ComposedType> rhs = checkComposedExpression(call.operands[1]);
        if (!rhs)
          return std::nullopt;
        if (rhs->elementType != lhs->elementType || rhs->extent != lhs->extent) {
          diagnostics.error(call.position, "binary elementwise builtins require operands of the "
                                           "same element type and extent");
          return std::nullopt;
        }
      }
      int64_t biasBound = int64_t(1) << fractionalBits;
      if (call.kind == ReductionKind::Offset &&
          (call.bias < -biasBound || call.bias > biasBound - 1)) {
        diagnostics.error(call.position, llvm::Twine("offset bias must be a raw signed Q1.") +
                                             llvm::Twine(fractionalBits) + " value in [" +
                                             llvm::Twine(-biasBound) + ", " +
                                             llvm::Twine(biasBound - 1) + "]");
        return std::nullopt;
      }
      if (call.kind == ReductionKind::Shift &&
          (call.amount < -fractionalBits || call.amount > fractionalBits)) {
        diagnostics.error(call.position, llvm::Twine("shift amount must lie in [-") +
                                             llvm::Twine(fractionalBits) + ", " +
                                             llvm::Twine(fractionalBits) + "]");
        return std::nullopt;
      }
      if (call.kind == ReductionKind::Div && (call.divisor < 1 || call.divisor > biasBound - 1)) {
        diagnostics.error(call.position, llvm::Twine("div divisor must be a positive integer in "
                                                     "[1, ") +
                                             llvm::Twine(biasBound - 1) + "]");
        return std::nullopt;
      }
      if (!checkBoundaryPolicies(call))
        return std::nullopt;
      if (call.kind == ReductionKind::Ratio) {
        // An unspelled policy is admitted only where the divisor's structure
        // proves it positive; otherwise the author names what a non-positive
        // divisor does, and no policy is ever chosen for them.
        if (call.nonpositive.empty()) {
          if (signFact(call.operands[1]) != SignFact::Positive) {
            diagnostics.error(call.position,
                              "the divisor is not provably positive (a magnitude or a square of "
                              "one operand plus a positive constant, under saturation); spell "
                              "ratio(x, y, nonpositive=trap) or nonpositive=saturate");
            return std::nullopt;
          }
        } else if (call.nonpositive != "trap" && call.nonpositive != "saturate") {
          diagnostics.error(call.position, llvm::Twine("unsupported nonpositive divisor policy '") +
                                               call.nonpositive + "'");
          return std::nullopt;
        }
      }
      return *lhs;
    };
    // The conversion is the one member whose result type is not its
    // operand's; the spelled direction and the spelled target must agree.
    auto checkConversion = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      std::optional<ComposedType> input = checkComposedExpression(call.operands.front());
      if (!input)
        return std::nullopt;
      bool fixedInput =
          input->elementType == SourceType::Q15 || input->elementType == SourceType::Q31;
      bool fixedTarget = call.target == SourceType::Q15 || call.target == SourceType::Q31;
      llvm::StringRef contract;
      switch (call.kind) {
      case ReductionKind::Widen:
        if (input->elementType != SourceType::Q15 || call.target != SourceType::Q31)
          contract = "widen takes a q15 operand to q31";
        break;
      case ReductionKind::Narrow:
        if (input->elementType != SourceType::Q31 || call.target != SourceType::Q15)
          contract = "narrow takes a q31 operand to q15";
        break;
      case ReductionKind::Quantize:
        if (input->elementType != SourceType::F32 || !fixedTarget)
          contract = "quantize takes an f32 operand to q15 or q31";
        break;
      default:
        if (!fixedInput || call.target != SourceType::F32)
          contract = "dequantize takes a q15 or q31 operand to f32";
        break;
      }
      if (!contract.empty()) {
        diagnostics.error(call.position, contract);
        return std::nullopt;
      }
      if (input->extent < 1 || input->extent > 4096) {
        diagnostics.error(call.position, "conversions currently require an extent in [1, 4096]");
        return std::nullopt;
      }
      if (isLossyConversionKind(call.kind) && !checkBoundaryPolicies(call))
        return std::nullopt;
      if (call.kind == ReductionKind::Quantize && call.destinationOverflow != "saturate") {
        diagnostics.error(call.position, "quantize saturates; wrapping an unbounded value is not "
                                         "a contract this operation offers");
        return std::nullopt;
      }
      return ComposedType{call.target, input->extent};
    };
    // The fixed gain: the same width in and out, the raw constant read at that
    // width, and the two tie rules its requantization admits.
    auto checkComposedGain = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      std::optional<ComposedType> input = checkComposedExpression(call.operands.front());
      if (!input)
        return std::nullopt;
      if (input->elementType != SourceType::Q15 && input->elementType != SourceType::Q31) {
        diagnostics.error(call.position, "gain requires q15 or q31 operand elements");
        return std::nullopt;
      }
      if (input->extent > 4096) {
        diagnostics.error(call.position, "gain currently requires an input extent in [1, 4096]");
        return std::nullopt;
      }
      unsigned width = input->elementType == SourceType::Q31 ? 32u : 16u;
      int64_t top = (int64_t(1) << (width - 1)) - 1;
      int64_t bottom = -(int64_t(1) << (width - 1));
      if (call.gain < bottom || call.gain > top) {
        diagnostics.error(call.position, llvm::Twine("gain constant must be a raw signed Q1.") +
                                             Twine(width - 1) + " value in [" + Twine(bottom) +
                                             ", " + Twine(top) + "]");
        return std::nullopt;
      }
      if (call.rounding.empty())
        call.rounding = "nearest_ties_positive";
      std::optional<ondsp::RoundingMode> parsed = parseRounding(call.rounding);
      if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven &&
                      *parsed != ondsp::RoundingMode::NearestTiesPositive)) {
        diagnostics.error(call.position,
                          "gain rounding must be nearest_even or nearest_ties_positive");
        return std::nullopt;
      }
      return *input;
    };
    // The kernel signature the call would have on its own, over its operand
    // parameters; a parameter in several slots gets one renamed copy per slot.
    auto checkLibraryValue = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      KernelAst library;
      library.name = ast.name;
      library.position = call.position;
      library.results.push_back(ResultTypeAst{SourceType::F32, false, {}});
      library.result = BuiltinCallAst(call);
      for (ExpressionAst &operand : library.result.operands) {
        const ParameterAst *parameter =
            operand.isParameterReference() ? parametersByName.lookup(operand.parameter) : nullptr;
        if (!parameter || parameter->type != SourceType::F32) {
          diagnostics.error(operand.position,
                            "a reduction inside an expression currently reads f32 parameters");
          return std::nullopt;
        }
        ++parameterUses[operand.parameter];
        ParameterAst copy = *parameter;
        if (llvm::any_of(library.parameters,
                         [&](const ParameterAst &other) { return other.name == copy.name; }))
          copy.name += "$" + std::to_string(library.parameters.size());
        operand.parameter = copy.name;
        library.parameters.push_back(std::move(copy));
      }
      std::optional<CheckedKernel> checked = checkKernel(std::move(library), diagnostics);
      if (!checked)
        return std::nullopt;
      call.library = std::make_shared<const CheckedKernel>(std::move(*checked));
      return ComposedType{SourceType::F32, 0};
    };
    checkComposedCall = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      if (call.kind == ReductionKind::Primitive)
        return checkPrimitive(call);
      if (isScalarReductionKind(call.kind))
        return checkLibraryValue(call);
      if (call.kind == ReductionKind::FirFilter)
        return checkNestedFirFilter(call);
      if (isElementwiseKind(call.kind))
        return checkElementwise(call);
      if (isConversionKind(call.kind))
        return checkConversion(call);
      if (call.kind == ReductionKind::Gain)
        return checkComposedGain(call);
      if (!isUnaryComposableKind(call.kind) || call.operands.size() != 1) {
        diagnostics.error(call.position,
                          "nested calls are currently supported only by unary FFT-family builtins, "
                          "a fir_filter input stage, and scalar dot/fir/rms");
        return std::nullopt;
      }
      std::optional<ComposedType> input = checkComposedExpression(call.operands.front());
      if (!input)
        return std::nullopt;

      if (call.kind == ReductionKind::ViterbiDecode) {
        if (input->elementType != SourceType::Q15) {
          diagnostics.error(call.position, "viterbi_decode requires q15 soft symbols");
          return std::nullopt;
        }
        int64_t constraint = call.constraintLength;
        int64_t rate = static_cast<int64_t>(call.polynomials.size());
        if (constraint < 3 || constraint > 7) {
          diagnostics.error(call.position, "viterbi_decode constraint_length must be in [3, 7]");
          return std::nullopt;
        }
        if (rate != 2 && rate != 3) {
          diagnostics.error(call.position,
                            "viterbi_decode requires two or three generator polynomials");
          return std::nullopt;
        }
        if (llvm::any_of(call.polynomials, [&](int64_t polynomial) {
              return polynomial < 1 || polynomial >= (int64_t{1} << constraint);
            })) {
          diagnostics.error(call.position, llvm::Twine("every generator polynomial must lie in [1, "
                                                       "2^constraint_length) = [1, ") +
                                               llvm::Twine(int64_t{1} << constraint) + ")");
          return std::nullopt;
        }
        if (input->extent % rate != 0 || (input->extent / rate) % 8 != 0 || input->extent > 16384) {
          diagnostics.error(call.position,
                            "viterbi_decode requires N * R symbols with N a positive multiple "
                            "of 8 and N * R <= 16384");
          return std::nullopt;
        }
        return ComposedType{SourceType::U8, input->extent / rate / 8};
      }
      if (call.kind == ReductionKind::Phase) {
        bool interleavedFp = input->elementType == SourceType::ComplexF32;
        if (input->elementType != SourceType::ComplexQ15 &&
            input->elementType != SourceType::ComplexQ31 && !interleavedFp) {
          diagnostics.error(
              call.position,
              "phase requires complex_q15, complex_q31, or complex_f32 operand elements");
          return std::nullopt;
        }
        if (input->extent < 1 || input->extent > 4096) {
          diagnostics.error(call.position,
                            "phase currently requires an operand extent in [1, 4096]");
          return std::nullopt;
        }
        if (interleavedFp) {
          if (!call.turn.empty()) {
            diagnostics.error(call.position,
                              "an f32 phase returns the format's own turn and names no width");
            return std::nullopt;
          }
          if (!parseFpContract(call.fpContract)) {
            diagnostics.error(call.position, "an f32 phase requires contract = off, fma, or fast");
            return std::nullopt;
          }
          return ComposedType{SourceType::F32, input->extent};
        }
        if (!call.fpContract.empty()) {
          diagnostics.error(call.position,
                            "a fixed-point phase declares no floating-point contract");
          return std::nullopt;
        }
        // The turn width is the call site's, independent of the component
        // width; the i32 storage of a Q0.32 turn is spelled q31 here.
        if (call.turn == "q31")
          return ComposedType{SourceType::Q31, input->extent};
        return ComposedType{SourceType::Q15, input->extent};
      }
      if (call.kind == ReductionKind::Power) {
        bool interleavedFp = input->elementType == SourceType::ComplexF32;
        if (input->elementType != SourceType::ComplexQ15 && !interleavedFp) {
          diagnostics.error(call.position,
                            "power requires complex_q15 or complex_f32 operand elements");
          return std::nullopt;
        }
        if (input->extent < 1 || input->extent > 4096) {
          diagnostics.error(call.position,
                            "power currently requires an operand extent in [1, 4096]");
          return std::nullopt;
        }
        if (interleavedFp) {
          if (!call.rounding.empty()) {
            diagnostics.error(call.position, "an f32 squared magnitude has no boundary to round");
            return std::nullopt;
          }
          if (!parseFpContract(call.fpContract)) {
            diagnostics.error(call.position,
                              "an f32 squared magnitude requires contract = off, fma, or fast");
            return std::nullopt;
          }
          return ComposedType{SourceType::F32, input->extent};
        }
        if (!call.fpContract.empty()) {
          diagnostics.error(call.position,
                            "a fixed-point squared magnitude declares no floating-point contract");
          return std::nullopt;
        }
        // The sum is exact at fraction 30 and the result is read at 15, so
        // this is the one boundary and it always exists.
        if (!call.rounding.empty()) {
          std::optional<ondsp::RoundingMode> parsed = parseRounding(call.rounding);
          if (!parsed || !isDeclaredExportRounding(*parsed)) {
            diagnostics.error(call.position,
                              "power rounding must be nearest_even, nearest_ties_positive, "
                              "toward_negative, or toward_zero");
            return std::nullopt;
          }
        }
        return ComposedType{SourceType::Q15, input->extent};
      }
      if (call.kind == ReductionKind::Magnitude) {
        bool complexQ31 = input->elementType == SourceType::ComplexQ31;
        bool interleavedFp = input->elementType == SourceType::ComplexF32;
        if (input->elementType != SourceType::ComplexQ15 && !complexQ31 && !interleavedFp) {
          diagnostics.error(
              call.position,
              "magnitude requires complex_q15, complex_q31, or complex_f32 operand elements");
          return std::nullopt;
        }
        if (input->extent < 1 || input->extent > 4096) {
          diagnostics.error(call.position,
                            "magnitude currently requires an operand extent in [1, 4096]");
          return std::nullopt;
        }
        if (interleavedFp) {
          if (!call.rounding.empty() || !call.inputRounding.empty()) {
            diagnostics.error(call.position, "an f32 magnitude has no boundary to round");
            return std::nullopt;
          }
          if (!parseFpContract(call.fpContract)) {
            diagnostics.error(call.position,
                              "an f32 magnitude requires contract = off, fma, or fast");
            return std::nullopt;
          }
          return ComposedType{SourceType::F32, input->extent};
        }
        if (!call.fpContract.empty()) {
          diagnostics.error(call.position,
                            "a fixed-point magnitude declares no floating-point contract");
          return std::nullopt;
        }
        if (!call.rounding.empty()) {
          std::optional<ondsp::RoundingMode> parsed = parseRounding(call.rounding);
          if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven &&
                          *parsed != ondsp::RoundingMode::TowardNegative)) {
            diagnostics.error(call.position,
                              "magnitude root_rounding must be nearest_even or toward_negative");
            return std::nullopt;
          }
        }
        // The component pre-shift exists only where the squares leave i64, so
        // naming its rounding at a width that has no such boundary is refused
        // rather than ignored.
        if (!call.inputRounding.empty()) {
          if (!complexQ31) {
            diagnostics.error(call.position,
                              "magnitude at this width has no component pre-shift to round");
            return std::nullopt;
          }
          std::optional<ondsp::RoundingMode> parsed = parseRounding(call.inputRounding);
          if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven &&
                          *parsed != ondsp::RoundingMode::TowardNegative)) {
            diagnostics.error(call.position,
                              "magnitude input_rounding must be nearest_even or toward_negative");
            return std::nullopt;
          }
        }
        return ComposedType{complexQ31 ? SourceType::Q31 : SourceType::Q15, input->extent};
      }
      // The interleaved floating-point profile covers all four transforms and
      // reads its extent the same way: complex operands count transform
      // points, an rfft real operand counts samples.
      bool floatingTransform =
          isFftKind(call.kind) &&
          (input->elementType == SourceType::ComplexF32 ||
           (call.kind == ReductionKind::Rfft && input->elementType == SourceType::F32));
      if (floatingTransform) {
        if (!call.rounding.empty() || !call.destinationOverflow.empty()) {
          diagnostics.error(
              call.position,
              "a floating-point transform has no stage boundary to round or saturate");
          return std::nullopt;
        }
        if (!parseFpContract(call.fpContract)) {
          diagnostics.error(call.position,
                            "a floating-point transform requires contract = off, fma, or fast");
          return std::nullopt;
        }
        int64_t points =
            call.kind == ReductionKind::Irfft ? (input->extent - 1) * 2 : input->extent;
        int64_t minimumPoints = isCfftKind(call.kind) ? 4 : 8;
        if (points < minimumPoints || points > 1024 || !llvm::isPowerOf2_64(points)) {
          diagnostics.error(call.position,
                            "a floating-point transform currently supports power-of-two sizes in "
                            "[4, 1024], and [8, 1024] for the real spellings");
          return std::nullopt;
        }
        if (call.kind == ReductionKind::Rfft)
          return ComposedType{SourceType::ComplexF32, points / 2 + 1};
        if (call.kind == ReductionKind::Irfft)
          return ComposedType{SourceType::F32, points};
        return *input;
      }
      if (isFftKind(call.kind) && !call.fpContract.empty()) {
        diagnostics.error(call.position,
                          "a fixed-point transform declares no floating-point contract");
        return std::nullopt;
      }
      if (isCfftKind(call.kind)) {
        if (input->elementType != SourceType::ComplexQ15 &&
            input->elementType != SourceType::ComplexQ31) {
          diagnostics.error(call.position,
                            "cfft and icfft require complex_q15 or complex_q31 operand elements");
          return std::nullopt;
        }
        if (input->elementType == SourceType::ComplexQ31) {
          if (input->extent < 4 || input->extent > 64 || !llvm::isPowerOf2_64(input->extent)) {
            diagnostics.error(
                call.position,
                "a complex_q31 cfft currently supports power-of-two extents in [4, 64]");
            return std::nullopt;
          }
          // The packed-Q31 profile has exactly one gated stage policy, so the
          // parameters admit only its spelling.
          if ((!call.rounding.empty() && parseRounding(call.rounding) !=
                                             std::optional(ondsp::RoundingMode::TowardNegative)) ||
              (!call.destinationOverflow.empty() &&
               parseOverflow(call.destinationOverflow) !=
                   std::optional(ondsp::OverflowMode::Saturate))) {
            diagnostics.error(call.position, "the packed-Q31 cfft profile is frozen to "
                                             "toward_negative saturating stages");
            return std::nullopt;
          }
          return *input;
        }
        if (input->extent != 4 && input->extent != 8) {
          diagnostics.error(call.position, "cfft currently supports only four or eight points");
          return std::nullopt;
        }
        std::optional<ondsp::RoundingMode> stageRounding =
            call.rounding.empty() ? std::optional(ondsp::RoundingMode::NearestEven)
                                  : parseRounding(call.rounding);
        std::optional<ondsp::OverflowMode> stageOverflow =
            call.destinationOverflow.empty() ? std::optional(ondsp::OverflowMode::Saturate)
                                             : parseOverflow(call.destinationOverflow);
        if (!stageRounding || *stageRounding == ondsp::RoundingMode::TowardZero) {
          diagnostics.error(call.position, "cfft rounding must be nearest_even, toward_negative, "
                                           "or nearest_ties_positive");
          return std::nullopt;
        }
        if (!stageOverflow) {
          diagnostics.error(call.position, "cfft overflow must be wrap or saturate");
          return std::nullopt;
        }
        if (*stageRounding == ondsp::RoundingMode::NearestEven &&
            *stageOverflow == ondsp::OverflowMode::Wrap) {
          diagnostics.error(call.position, "a nearest_even cfft requires saturating overflow");
          return std::nullopt;
        }
        return *input;
      }
      if (call.kind == ReductionKind::Rfft) {
        if (input->elementType != SourceType::Q15 && input->elementType != SourceType::Q31) {
          diagnostics.error(call.position, "rfft requires Q15 or Q31 real operand elements");
          return std::nullopt;
        }
        if (input->extent < 8 || input->extent > 64 || !llvm::isPowerOf2_64(input->extent)) {
          diagnostics.error(call.position,
                            "rfft currently supports power-of-two extents in [8, 64]");
          return std::nullopt;
        }
        return ComposedType{input->elementType == SourceType::Q15 ? SourceType::ComplexQ15
                                                                  : SourceType::ComplexQ31,
                            input->extent / 2 + 1};
      }
      if (input->elementType == SourceType::ComplexQ31) {
        int64_t realExtent = (input->extent - 1) * 2;
        if (realExtent < 8 || realExtent > 64 || !llvm::isPowerOf2_64(realExtent)) {
          diagnostics.error(call.position, "a complex_q31 irfft currently supports Hermitian bin "
                                           "counts for power-of-two extents in [8, 64]");
          return std::nullopt;
        }
        return ComposedType{SourceType::Q31, realExtent};
      }
      if (input->elementType != SourceType::ComplexQ15) {
        diagnostics.error(call.position,
                          "irfft requires complex_q15 or complex_q31 Hermitian operand elements");
        return std::nullopt;
      }
      if (input->extent != 5 && input->extent != 9) {
        diagnostics.error(call.position,
                          "irfft currently supports only five or nine Hermitian bins");
        return std::nullopt;
      }
      return ComposedType{SourceType::Q15, (input->extent - 1) * 2};
    };

    std::optional<ComposedType> inferred = checkComposedCall(ast.result);
    if (!inferred)
      return std::nullopt;
    // Reading a parameter more than once is fine: a tensor operand is a
    // value, so `mult(x, x)` reads the same storage twice and writes neither.
    for (const ParameterAst &parameter : ast.parameters) {
      if (parameterUses.lookup(parameter.name) == 0) {
        diagnostics.error(parameter.position, llvm::Twine("parameter '") + parameter.name +
                                                  "' is never consumed by the kernel expression");
        return std::nullopt;
      }
    }
    if (!ast.primaryResult().tensor && ast.result.kind == ReductionKind::Primitive) {
      if (inferred->extent != 0 || ast.primaryResult().type != inferred->elementType) {
        diagnostics.error(ast.result.position,
                          "declared result type does not match the expression");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast)};
    }
    if (!ast.primaryResult().tensor || !hasRank(ast.primaryResult().shape, 1)) {
      diagnostics.error(ast.result.position,
                        "composable builtins currently require a rank-1 tensor result");
      return std::nullopt;
    }
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    if (!resultExtent) {
      diagnostics.error(ast.result.position,
                        "composable builtins currently require a static result extent");
      return std::nullopt;
    }
    if (ast.primaryResult().type != inferred->elementType || *resultExtent != inferred->extent) {
      diagnostics.error(ast.result.position,
                        "declared result type does not match the builtin expression");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, std::nullopt};
  }

  if (ast.primaryResult().type == SourceType::ComplexQ15) {
    diagnostics.error(
        ast.result.position,
        "complex_q15 is currently supported only by FFT-family and butterfly builtins");
    return std::nullopt;
  }
  if (ast.primaryResult().type == SourceType::ComplexQ31) {
    diagnostics.error(ast.result.position,
                      "complex_q31 is currently supported only by the cfft, icfft, rfft, and "
                      "irfft builtins and the complex reductions");
    return std::nullopt;
  }
  if (ast.primaryResult().type == SourceType::ComplexF32) {
    diagnostics.error(ast.result.position, "complex_f32 is currently supported only by the cfft, "
                                           "icfft, and rfft builtins");
    return std::nullopt;
  }

  if (constexprCount != 0) {
    if (constexprCount != 1 || !rhsParameter || !rhsParameter->isConstexpr() || !lhsParameter ||
        lhsParameter->isConstexpr()) {
      diagnostics.error(
          ast.result.position,
          "constexpr is supported only for the right operand of a fixed-point reduction");
      return std::nullopt;
    }
    if (ast.primaryResult().type == SourceType::F32) {
      diagnostics.error(ast.result.position,
                        "constexpr parameters are restricted to fixed-point FIR coefficients");
      return std::nullopt;
    }
    if (rhsParameter->constantValues.empty()) {
      diagnostics.error(rhsParameter->position, "constexpr reduction operand cannot be empty");
      return std::nullopt;
    }
    int64_t minimum = ast.primaryResult().type == SourceType::Q15
                          ? static_cast<int64_t>(std::numeric_limits<int16_t>::min())
                          : static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    int64_t maximum = ast.primaryResult().type == SourceType::Q15
                          ? static_cast<int64_t>(std::numeric_limits<int16_t>::max())
                          : static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    for (int64_t value : rhsParameter->constantValues) {
      if (value < minimum || value > maximum) {
        diagnostics.error(rhsParameter->position,
                          ast.primaryResult().type == SourceType::Q15
                              ? "Q15 constexpr coefficient is outside signed i16 storage range"
                              : "Q31 constexpr coefficient is outside signed i32 storage range");
        return std::nullopt;
      }
    }
  }

  bool isConv1D = ast.result.kind == ReductionKind::Convolution ||
                  ast.result.kind == ReductionKind::Correlation;
  bool isFirDecimate = ast.result.kind == ReductionKind::FirDecimate;
  bool isFirInterpolate = ast.result.kind == ReductionKind::FirInterpolate;
  if (isConv1D) {
    if (constexprCount != 0) {
      diagnostics.error(ast.result.position,
                        "convolution and correlation currently require runtime tensor operands");
      return std::nullopt;
    }
    if (!ast.primaryResult().tensor || !lhsParameter || !rhsParameter ||
        !lhsParameter->isTensor() || !rhsParameter->isTensor()) {
      diagnostics.error(ast.result.position,
                        "convolution and correlation require tensor inputs and result");
      return std::nullopt;
    }
    const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &rhsExtent = getRankOneExtent(rhsParameter->shape);
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    if ((!lhsExtent || !rhsExtent) && resultExtent) {
      diagnostics.error(ast.result.position,
                        "a static convolution/correlation result requires static input and kernel "
                        "extents");
      return std::nullopt;
    }
    if (lhsExtent && rhsExtent) {
      if (*rhsExtent > *lhsExtent) {
        diagnostics.error(ast.result.position,
                          "convolution/correlation input extent must cover the kernel");
        return std::nullopt;
      }
      int64_t expectedExtent = *lhsExtent - *rhsExtent + 1;
      if (resultExtent && *resultExtent != expectedExtent) {
        diagnostics.error(ast.result.position,
                          "static convolution/correlation result extent is incorrect");
        return std::nullopt;
      }
    }
  } else if (isFirDecimate) {
    if (constexprCount != 0 || !ast.primaryResult().tensor ||
        (ast.primaryResult().type != SourceType::Q15 &&
         ast.primaryResult().type != SourceType::Q31 &&
         ast.primaryResult().type != SourceType::F32) ||
        !lhsParameter || !rhsParameter || !lhsParameter->isTensor() || !rhsParameter->isTensor()) {
      diagnostics.error(
          ast.result.position,
          "fir_decimate requires Q15, Q31 or f32 tensor input, coefficients, and result");
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 1) ||
        !hasRank(ast.primaryResult().shape, 1)) {
      diagnostics.error(ast.result.position, "fir_decimate currently requires rank-1 tensors");
      return std::nullopt;
    }
    if (ast.result.factor != 2) {
      diagnostics.error(ast.result.position,
                        "fir_decimate source binding currently requires factor=2");
      return std::nullopt;
    }
    const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &rhsExtent = getRankOneExtent(rhsParameter->shape);
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    if (!lhsExtent || !rhsExtent || !resultExtent) {
      diagnostics.error(ast.result.position,
                        "fir_decimate source binding currently requires static extents");
      return std::nullopt;
    }
    if (*rhsExtent > *lhsExtent) {
      diagnostics.error(ast.result.position,
                        "fir_decimate input extent must cover the coefficient window");
      return std::nullopt;
    }
    int64_t expectedExtent = (*lhsExtent - *rhsExtent) / ast.result.factor + 1;
    if (*resultExtent != expectedExtent) {
      diagnostics.error(ast.result.position, "static fir_decimate result extent is incorrect");
      return std::nullopt;
    }
  } else if (isFirInterpolate) {
    if (constexprCount != 0 || !ast.primaryResult().tensor ||
        (ast.primaryResult().type != SourceType::Q15 &&
         ast.primaryResult().type != SourceType::F32) ||
        !lhsParameter || !rhsParameter || !lhsParameter->isTensor() || !rhsParameter->isTensor()) {
      diagnostics.error(
          ast.result.position,
          "fir_interpolate requires Q15 or f32 tensor input, coefficients, and result");
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 1) || !hasRank(rhsParameter->shape, 1) ||
        !hasRank(ast.primaryResult().shape, 1)) {
      diagnostics.error(ast.result.position, "fir_interpolate currently requires rank-1 tensors");
      return std::nullopt;
    }
    if (ast.result.factor != 2) {
      diagnostics.error(ast.result.position,
                        "fir_interpolate source binding currently requires factor=2");
      return std::nullopt;
    }
    const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &rhsExtent = getRankOneExtent(rhsParameter->shape);
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    if (!lhsExtent || !rhsExtent || !resultExtent) {
      diagnostics.error(ast.result.position,
                        "fir_interpolate source binding currently requires static extents");
      return std::nullopt;
    }
    int64_t inputIntervals = *lhsExtent - 1;
    if (inputIntervals > (std::numeric_limits<int64_t>::max() - *rhsExtent) / ast.result.factor) {
      diagnostics.error(ast.result.position, "fir_interpolate result extent overflows index");
      return std::nullopt;
    }
    int64_t expectedExtent = inputIntervals * ast.result.factor + *rhsExtent;
    if (*resultExtent != expectedExtent) {
      diagnostics.error(ast.result.position, "static fir_interpolate result extent is incorrect");
      return std::nullopt;
    }
  } else if (ast.result.kind == ReductionKind::FirFilter) {
    if (!ast.primaryResult().tensor) {
      diagnostics.error(ast.result.position, "fir_filter must return a tensor value");
      return std::nullopt;
    }
    if (!lhsParameter || !rhsParameter || !lhsParameter->isTensor() ||
        (!rhsParameter->isTensor() && !rhsParameter->isConstexpr())) {
      diagnostics.error(ast.result.position,
                        "fir_filter currently requires tensor input and tensor or constexpr "
                        "coefficients");
      return std::nullopt;
    }
    const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    std::optional<int64_t> coefficientExtent =
        rhsParameter->isConstexpr() ? std::nullopt : getRankOneExtent(rhsParameter->shape);
    if (rhsParameter->isConstexpr())
      coefficientExtent = static_cast<int64_t>(rhsParameter->constantValues.size());
    if (ast.result.boundary == "valid") {
      if ((!lhsExtent || !coefficientExtent) && resultExtent) {
        diagnostics.error(ast.result.position,
                          "a static valid fir_filter result requires static input and coefficient "
                          "extents");
        return std::nullopt;
      }
      if (lhsExtent && coefficientExtent) {
        if (*coefficientExtent > *lhsExtent) {
          diagnostics.error(ast.result.position,
                            "valid fir_filter requires input extent at least coefficient extent");
          return std::nullopt;
        }
        int64_t expectedExtent = *lhsExtent - *coefficientExtent + 1;
        if (resultExtent && *resultExtent != expectedExtent) {
          diagnostics.error(ast.result.position,
                            "static fir_filter result extent does not match valid convolution");
          return std::nullopt;
        }
      }
    } else if (ast.result.boundary == "full") {
      if (!lhsExtent || !coefficientExtent || !resultExtent) {
        diagnostics.error(ast.result.position,
                          "full fir_filter currently requires static input, coefficient, and "
                          "result extents");
        return std::nullopt;
      }
      if (*lhsExtent > std::numeric_limits<int64_t>::max() - (*coefficientExtent - 1)) {
        diagnostics.error(ast.result.position, "full fir_filter result extent overflows index");
        return std::nullopt;
      }
      int64_t expectedExtent = *lhsExtent + *coefficientExtent - 1;
      if (*resultExtent != expectedExtent) {
        diagnostics.error(ast.result.position,
                          "static fir_filter result extent does not match full convolution");
        return std::nullopt;
      }
    } else {
      diagnostics.error(ast.result.position,
                        "fir_filter supports only boundary=valid or boundary=full");
      return std::nullopt;
    }
  } else if (isUnaryTensorKind(ast.result.kind)) {
    llvm::StringRef builtin = ast.result.kind == ReductionKind::Dct             ? "dct"
                              : ast.result.kind == ReductionKind::MovingAverage ? "moving_average"
                              : ast.result.kind == ReductionKind::Gain          ? "gain"
                              : ast.result.kind == ReductionKind::Rms           ? "rms"
                              : ast.result.kind == ReductionKind::Sine          ? "sine"
                              : ast.result.kind == ReductionKind::CicDecimate   ? "cic_decimate"
                              : ast.result.kind == ReductionKind::Log2          ? "log2"
                              : ast.result.kind == ReductionKind::Exp2          ? "exp2"
                              : ast.result.kind == ReductionKind::Goertzel      ? "goertzel"
                                                                                : "cosine";
    bool admitsFloat =
        ast.result.kind == ReductionKind::Rms || ast.result.kind == ReductionKind::MovingAverage ||
        ast.result.kind == ReductionKind::Dct || ast.result.kind == ReductionKind::Gain ||
        ast.result.kind == ReductionKind::Goertzel;
    bool isFloat = ast.primaryResult().type == SourceType::F32;
    // Every fixed-point member here carries a Q31 profile.
    bool admitsQ31 =
        ast.result.kind == ReductionKind::Rms || ast.result.kind == ReductionKind::Dct ||
        ast.result.kind == ReductionKind::Gain || ast.result.kind == ReductionKind::MovingAverage ||
        ast.result.kind == ReductionKind::CicDecimate || ast.result.kind == ReductionKind::Sine ||
        ast.result.kind == ReductionKind::Cosine || ast.result.kind == ReductionKind::Log2 ||
        ast.result.kind == ReductionKind::Exp2;
    bool isQ31 = ast.primaryResult().type == SourceType::Q31;
    // The Q15 goertzel energy is tensor<1xi64>, a storage width no source
    // type names, so only the f32 profile has a spelling here.
    if (ast.result.kind == ReductionKind::Goertzel && !isFloat) {
      diagnostics.error(ast.result.position, "goertzel currently binds only the f32 profile");
      return std::nullopt;
    }
    // The scalar rms spelling mirrors scalar dot: one buffer operand reduced
    // to a bare f32, so the boundary owns no allocation.
    if (ast.result.kind == ReductionKind::Rms && !ast.primaryResult().tensor) {
      if (!isFloat || constexprCount != 0 || !lhsParameter || lhsParameter->isTensor() ||
          lhsParameter->isScalar()) {
        diagnostics.error(ast.result.position, "scalar rms requires one f32 buffer operand");
        return std::nullopt;
      }
      const std::optional<int64_t> &extent = getRankOneExtent(lhsParameter->shape);
      if (!hasRank(lhsParameter->shape, 1) || !extent || *extent < 2 || *extent > 4096) {
        diagnostics.error(ast.result.position,
                          "rms currently requires an input extent in [2, 4096]");
        return std::nullopt;
      }
      auto contract = parseFpContract(ast.result.fpContract);
      if (!contract) {
        diagnostics.error(ast.result.position,
                          llvm::Twine("unsupported floating-point contract '") +
                              ast.result.fpContract + "'");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
    }
    if (constexprCount != 0 || !ast.primaryResult().tensor || !lhsParameter ||
        !lhsParameter->isTensor() ||
        (ast.primaryResult().type != SourceType::Q15 && !(admitsFloat && isFloat) &&
         !(admitsQ31 && isQ31))) {
      diagnostics.error(ast.result.position,
                        llvm::Twine(builtin) +
                            (admitsQ31     ? " requires a Q15, Q31 or f32 tensor input and a "
                                             "matching result"
                             : admitsFloat ? " requires a Q15 or f32 tensor input "
                                             "and a matching result"
                                           : " requires a Q15 tensor input and "
                                             "result"));
      return std::nullopt;
    }
    if (!hasRank(lhsParameter->shape, 1) || !hasRank(ast.primaryResult().shape, 1)) {
      diagnostics.error(ast.result.position,
                        llvm::Twine(builtin) + " currently requires rank-1 tensors");
      return std::nullopt;
    }
    const std::optional<int64_t> &inputExtent = getRankOneExtent(lhsParameter->shape);
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    if (!inputExtent || !resultExtent) {
      diagnostics.error(ast.result.position,
                        llvm::Twine(builtin) + " currently requires static extents");
      return std::nullopt;
    }
    if (ast.result.kind == ReductionKind::Dct) {
      // The source type system only names the storage: the declared result
      // reads as q15 or q31 while the emitted operation carries the derived
      // frac = W - 2 - log2(N) output reading in its attribute. The
      // projection is lossy but currently safe — dct does not compose, so
      // no in-language consumer can misread the scale.
      if (*inputExtent < 4 || *inputExtent > 64 || !llvm::isPowerOf2_64(*inputExtent)) {
        diagnostics.error(ast.result.position,
                          "dct currently requires a power-of-two input extent in [4, 64]");
        return std::nullopt;
      }
      if (*resultExtent != *inputExtent) {
        diagnostics.error(ast.result.position, "dct result extent must equal the input extent");
        return std::nullopt;
      }
    } else if (ast.result.kind == ReductionKind::MovingAverage) {
      int64_t window = ast.result.window;
      if (window < 2 || window > 64) {
        diagnostics.error(ast.result.position,
                          "moving_average currently requires a window in [2, 64]");
        return std::nullopt;
      }
      if (window > *inputExtent) {
        diagnostics.error(ast.result.position, "moving_average input extent must cover the window");
        return std::nullopt;
      }
      if (*resultExtent != *inputExtent - window + 1) {
        diagnostics.error(ast.result.position, "static moving_average result extent is incorrect");
        return std::nullopt;
      }
    } else if (ast.result.kind == ReductionKind::Gain) {
      if (isFloat) {
        if (ast.result.literal) {
          diagnostics.error(ast.result.position,
                            "an integer literal is a raw fixed-point value; "
                            "scale f32 with gain(x, gain=[n, d], contract=...)");
          return std::nullopt;
        }
        auto [constant, refusal] = roundRationalToF32(ast.result.gain, ast.result.fpConstantDen);
        if (refusal != RationalRefusal::None) {
          diagnostics.error(ast.result.position,
                            llvm::Twine("f32 gain constant: ") + describeRationalRefusal(refusal));
          return std::nullopt;
        }
        ast.result.fpConstant = constant;
      } else {
        // The raw constant is read at the declared width, so a Q31 gain names
        // a Q1.31 integer rather than a Q1.15 one.
        unsigned width = isQ31 ? 32u : 16u;
        int64_t top = (int64_t(1) << (width - 1)) - 1;
        int64_t bottom = -(int64_t(1) << (width - 1));
        if (ast.result.gain < bottom || ast.result.gain > top) {
          diagnostics.error(ast.result.position,
                            llvm::Twine("gain constant must be a raw signed Q1.") +
                                Twine(width - 1) + " value in [" + Twine(bottom) + ", " +
                                Twine(top) + "]");
          return std::nullopt;
        }
      }
      if (*inputExtent > 4096) {
        diagnostics.error(ast.result.position,
                          "gain currently requires an input extent in [1, 4096]");
        return std::nullopt;
      }
      if (*resultExtent != *inputExtent) {
        diagnostics.error(ast.result.position, "gain result extent must equal the input extent");
        return std::nullopt;
      }
    } else if (ast.result.kind == ReductionKind::CicDecimate) {
      int64_t stages = ast.result.stages;
      int64_t rate = ast.result.rate;
      int64_t delay = ast.result.delay;
      if (stages < 1 || stages > 8) {
        diagnostics.error(ast.result.position, "cic_decimate requires stages in [1, 8]");
        return std::nullopt;
      }
      if (rate < 2 || rate > 4096 || !llvm::isPowerOf2_64(uint64_t(rate))) {
        diagnostics.error(ast.result.position,
                          "cic_decimate requires a power-of-two rate in [2, 4096]");
        return std::nullopt;
      }
      if (delay != 1 && delay != 2) {
        diagnostics.error(ast.result.position,
                          "cic_decimate requires a differential delay of 1 or 2");
        return std::nullopt;
      }
      if (16 + stages * int64_t(llvm::Log2_64(uint64_t(rate * delay))) > 64) {
        diagnostics.error(ast.result.position,
                          "cic_decimate requires stages * log2(rate * delay) <= 48");
        return std::nullopt;
      }
      if (*resultExtent < 1 || *resultExtent > 4096 || *inputExtent != *resultExtent * rate) {
        diagnostics.error(ast.result.position,
                          "cic_decimate input extent must be the rate times the result extent");
        return std::nullopt;
      }
    } else if (ast.result.kind == ReductionKind::Rms) {
      if (*inputExtent < 2 || *inputExtent > 4096 ||
          (!isFloat && !llvm::isPowerOf2_64(*inputExtent))) {
        diagnostics.error(ast.result.position,
                          isFloat ? "rms currently requires an input extent in [2, 4096]"
                                  : "rms currently requires a power-of-two input extent in "
                                    "[2, 4096]");
        return std::nullopt;
      }
      if (*resultExtent != 1) {
        diagnostics.error(ast.result.position, "rms returns a single-element tensor");
        return std::nullopt;
      }
    } else if (ast.result.kind == ReductionKind::Goertzel) {
      if (*inputExtent < 2 || *inputExtent > 4096) {
        diagnostics.error(ast.result.position,
                          "goertzel currently requires an input extent in [2, 4096]");
        return std::nullopt;
      }
      if (*resultExtent != 1) {
        diagnostics.error(ast.result.position, "goertzel returns a single-element tensor");
        return std::nullopt;
      }
      if (ast.result.bin < 0 || ast.result.bin > *inputExtent / 2) {
        diagnostics.error(ast.result.position, "goertzel bin must lie in [0, N/2]");
        return std::nullopt;
      }
    } else {
      if (*inputExtent > 4096) {
        diagnostics.error(ast.result.position,
                          llvm::Twine(builtin) +
                              " currently requires an input extent in [1, 4096]");
        return std::nullopt;
      }
      if (*resultExtent != *inputExtent) {
        diagnostics.error(ast.result.position,
                          llvm::Twine(builtin) + " result extent must equal the input extent");
        return std::nullopt;
      }
    }
    // Omission keeps the contract default: requantization boundaries (gain,
    // cic_decimate) take the export default, algorithm-internal roots stay
    // nearest_even.
    bool isGain = ast.result.kind == ReductionKind::Gain;
    bool isCic = ast.result.kind == ReductionKind::CicDecimate;
    bool isDct = ast.result.kind == ReductionKind::Dct;
    ondsp::RoundingMode rounding = isGain || isCic ? ondsp::RoundingMode::NearestTiesPositive
                                                   : ondsp::RoundingMode::NearestEven;
    if (!ast.result.rounding.empty() && isDct) {
      std::optional<ondsp::RoundingMode> parsed = parseRounding(ast.result.rounding);
      if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven &&
                      *parsed != ondsp::RoundingMode::TowardNegative &&
                      *parsed != ondsp::RoundingMode::NearestTiesPositive)) {
        diagnostics.error(ast.result.position, "dct rounding must be nearest_even, "
                                               "toward_negative, or nearest_ties_positive");
        return std::nullopt;
      }
      rounding = *parsed;
    } else if (!ast.result.rounding.empty()) {
      ondsp::RoundingMode alternative = isGain || isCic ? ondsp::RoundingMode::NearestTiesPositive
                                                        : ondsp::RoundingMode::TowardNegative;
      std::optional<ondsp::RoundingMode> parsed = parseRounding(ast.result.rounding);
      if (!parsed || (*parsed != ondsp::RoundingMode::NearestEven && *parsed != alternative)) {
        diagnostics.error(
            ast.result.position,
            isGain  ? "gain rounding must be nearest_even or nearest_ties_positive"
            : isCic ? "cic_decimate rounding must be nearest_even or nearest_ties_positive"
                    : "rms root_rounding must be nearest_even or toward_negative");
        return std::nullopt;
      }
      rounding = *parsed;
    }
    if (ast.result.kind == ReductionKind::CicDecimate) {
      std::optional<ondsp::OverflowMode> overflow = parseOverflow(ast.result.stateOverflow);
      if (!overflow) {
        diagnostics.error(ast.result.position, llvm::Twine("unsupported state overflow mode '") +
                                                   ast.result.stateOverflow + "'");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast), std::nullopt, rounding, std::nullopt,
                           std::nullopt,   std::nullopt, *overflow};
    }
    if (isFloat) {
      auto contract = parseFpContract(ast.result.fpContract);
      if (!contract) {
        diagnostics.error(ast.result.position,
                          llvm::Twine("unsupported floating-point contract '") +
                              ast.result.fpContract + "'");
        return std::nullopt;
      }
      return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
    }
    // The pre-shift boundary exists only where an exact sum of squares would
    // not fit i64, so the binding declares its rounding exactly there and
    // refuses it everywhere else.
    std::optional<ondsp::RoundingMode> inputRounding;
    std::optional<ondsp::RoundingMode> productRounding;
    // dct rounds a PRODUCT where rms rounds an INPUT, so the shared source
    // slot resolves to different checked fields and different admitted modes.
    bool rawHigh = isDct && ast.result.product == "raw_high";
    if (isDct && !ast.result.product.empty() && ast.result.product != "full" && !rawHigh) {
      diagnostics.error(ast.result.position, llvm::Twine("unsupported product selection '") +
                                                 ast.result.product + "'; use full or raw_high");
      return std::nullopt;
    }
    if (rawHigh && !isQ31) {
      diagnostics.error(ast.result.position,
                        "product=raw_high is the Q31 dct profile: each term is the raw high half "
                        "of the product, accumulated at frac 30");
      return std::nullopt;
    }
    if (rawHigh && !ast.result.inputRounding.empty()) {
      diagnostics.error(ast.result.position,
                        "a raw-high dct has no product rounding to declare: the high half is a "
                        "floor");
      return std::nullopt;
    }
    bool hasProductShift = isDct && !isFloat && !rawHigh &&
                           ir::getReductionProductShift(isQ31 ? 32 : 16, *inputExtent) > 0;
    bool hasPreShift = ast.result.kind == ReductionKind::Rms && !isFloat &&
                       ir::getRmsInputPreShift(isQ31 ? 32 : 16, *inputExtent) > 0;
    if (!ast.result.inputRounding.empty()) {
      if (!hasPreShift && !hasProductShift) {
        diagnostics.error(ast.result.position,
                          isDct
                              ? "dct at this width and extent has no product boundary to round"
                              : "rms at this width and extent has no pre-shift boundary to round");
        return std::nullopt;
      }
      std::optional<ondsp::RoundingMode> parsed = parseRounding(ast.result.inputRounding);
      bool admitted = parsed && (*parsed == ondsp::RoundingMode::NearestEven ||
                                 *parsed == ondsp::RoundingMode::TowardNegative ||
                                 (isDct && *parsed == ondsp::RoundingMode::NearestTiesPositive));
      if (!admitted) {
        diagnostics.error(ast.result.position,
                          isDct ? "dct product_rounding must be nearest_even, toward_negative, or "
                                  "nearest_ties_positive"
                                : "rms input_rounding must be nearest_even or toward_negative");
        return std::nullopt;
      }
      (isDct ? productRounding : inputRounding) = *parsed;
    } else if (hasPreShift) {
      inputRounding = ondsp::RoundingMode::NearestEven;
    } else if (hasProductShift) {
      productRounding = ondsp::RoundingMode::NearestEven;
    }
    CheckedKernel checked{std::move(ast), std::nullopt, rounding, std::nullopt, std::nullopt};
    checked.inputRounding = inputRounding;
    checked.productRounding = productRounding;
    return checked;
  } else {
    if (ast.primaryResult().tensor) {
      diagnostics.error(ast.result.position, "dot and fir return scalar values");
      return std::nullopt;
    }
    if (llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return parameter.isTensor(); })) {
      diagnostics.error(ast.result.position,
                        "scalar dot and fir currently require buffer operands");
      return std::nullopt;
    }
  }

  if (ast.primaryResult().type == SourceType::F32) {
    auto contract = parseFpContract(ast.result.fpContract);
    if (!contract) {
      diagnostics.error(ast.result.position, llvm::Twine("unsupported floating-point contract '") +
                                                 ast.result.fpContract + "'");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
  }

  if (constexprCount != 0) {
    if (ast.result.kind != ReductionKind::FirFilter) {
      const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
      if (!lhsExtent) {
        diagnostics.error(lhsParameter->position,
                          "a constexpr reduction operand requires a static left operand extent");
        return std::nullopt;
      }
      if (*lhsExtent != static_cast<int64_t>(rhsParameter->constantValues.size())) {
        diagnostics.error(lhsParameter->position,
                          "static input extent must equal the constexpr coefficient count");
        return std::nullopt;
      }
    }
  }

  if (int64_t gain = ast.result.gainBound) {
    if (!rhsParameter->isBuffer() && !rhsParameter->isConstexpr()) {
      diagnostics.error(rhsParameter->position,
                        "gain_bound declares a precondition on a buffer or constexpr coefficient "
                        "operand");
      return std::nullopt;
    }
    // A constant table's gain is known here, so the promise is checked now.
    if (rhsParameter->isConstexpr()) {
      int64_t frac = ast.primaryResult().type == SourceType::Q31 ? 31 : 15;
      int64_t sum = 0;
      for (int64_t value : rhsParameter->constantValues)
        sum += value < 0 ? -value : value;
      if (sum >= (gain << frac)) {
        diagnostics.error(rhsParameter->position,
                          llvm::Twine("the constexpr coefficients' absolute sum ") +
                              llvm::Twine(sum) + " is not below gain_bound=" + llvm::Twine(gain));
        return std::nullopt;
      }
    }
  }

  if (ast.result.accumulatorAuto) {
    bool isScalarReduction =
        ast.result.kind == ReductionKind::Dot || ast.result.kind == ReductionKind::Fir;
    // Every window of a full-boundary fir_filter overlaps at most K taps, so
    // the valid-boundary bound covers both boundaries unchanged.
    bool isWindowReduction = ast.result.kind == ReductionKind::Convolution ||
                             ast.result.kind == ReductionKind::Correlation ||
                             ast.result.kind == ReductionKind::FirDecimate ||
                             ast.result.kind == ReductionKind::FirInterpolate ||
                             ast.result.kind == ReductionKind::FirFilter;
    if (ast.primaryResult().type != SourceType::Q15 || (!isScalarReduction && !isWindowReduction)) {
      diagnostics.error(
          ast.result.position,
          "automatic accumulation currently supports static Q15 dot, fir, fir_filter, "
          "fir_decimate, fir_interpolate, and conv1d");
      return std::nullopt;
    }
    std::optional<int64_t> rhsExtent =
        rhsParameter->isConstexpr() ? std::nullopt : getRankOneExtent(rhsParameter->shape);
    if (rhsParameter->isConstexpr())
      rhsExtent = static_cast<int64_t>(rhsParameter->constantValues.size());
    if (!rhsExtent) {
      diagnostics.error(ast.result.position,
                        isScalarReduction
                            ? "automatic accumulation requires equal static operand extents"
                            : "automatic accumulation requires a static coefficient extent");
      return std::nullopt;
    }
    const std::optional<int64_t> &lhsExtent = getRankOneExtent(lhsParameter->shape);
    if (isScalarReduction && (!lhsExtent || *lhsExtent != *rhsExtent)) {
      diagnostics.error(ast.result.position,
                        "automatic accumulation requires equal static operand extents");
      return std::nullopt;
    }
    uint64_t productCount = static_cast<uint64_t>(*rhsExtent);
    if (isFirInterpolate)
      productCount = (productCount + static_cast<uint64_t>(ast.result.factor) - 1) /
                     static_cast<uint64_t>(ast.result.factor);
    ast.result.accumulatorWidth = inferQ15FullAccumulatorWidth(productCount);
  }

  // The raw high half is the Q31 target's native product: each term keeps the
  // upper 32 bits of the exact product at frac 30, so the accumulator is the
  // shared i40/frac30 one and the readout doubles once to reach Q31.
  bool rawHigh = ast.result.product == "raw_high";
  if (!ast.result.product.empty() && ast.result.product != "full" && !rawHigh) {
    diagnostics.error(ast.result.position, llvm::Twine("unsupported product selection '") +
                                               ast.result.product + "'; use full or raw_high");
    return std::nullopt;
  }
  if (rawHigh &&
      (ast.primaryResult().type != SourceType::Q31 ||
       (ast.result.kind != ReductionKind::Dot && ast.result.kind != ReductionKind::Fir))) {
    diagnostics.error(ast.result.position,
                      "product=raw_high is the Q31 dot and fir profile: each term is the raw high "
                      "half of the product, accumulated at frac 30");
    return std::nullopt;
  }
  uint64_t requiredAccumulatorWidth =
      ast.primaryResult().type == SourceType::Q15 || rawHigh ? 40 : 64;
  if (!ast.result.accumulatorAuto && ast.result.accumulatorWidth != requiredAccumulatorWidth) {
    diagnostics.error(ast.result.position,
                      ast.primaryResult().type == SourceType::Q15
                          ? "the executable Q15 profile requires exact accumulator width 40"
                      : rawHigh ? "the executable Q31 raw-high profile requires exact accumulator "
                                  "width 40"
                                : "the executable Q31 profile requires exact accumulator width 64");
    return std::nullopt;
  }

  auto updateOverflow = parseOverflow(ast.result.updateOverflow);
  if (!updateOverflow) {
    diagnostics.error(ast.result.position, llvm::Twine("unsupported update overflow mode '") +
                                               ast.result.updateOverflow + "'");
    return std::nullopt;
  }
  auto rounding = parseRounding(ast.result.rounding);
  if (!rounding) {
    diagnostics.error(ast.result.position,
                      llvm::Twine("unsupported rounding mode '") + ast.result.rounding + "'");
    return std::nullopt;
  }
  if (!isDeclaredExportRounding(*rounding)) {
    diagnostics.error(
        ast.result.position,
        "export rounding must be nearest_even, nearest_ties_positive, toward_negative, or "
        "toward_zero");
    return std::nullopt;
  }
  auto destinationOverflow = parseOverflow(ast.result.destinationOverflow);
  if (!destinationOverflow) {
    diagnostics.error(ast.result.position, llvm::Twine("unsupported destination overflow mode '") +
                                               ast.result.destinationOverflow + "'");
    return std::nullopt;
  }

  return CheckedKernel{std::move(ast), *updateOverflow, *rounding, *destinationOverflow,
                       std::nullopt};
}

static Location getLocation(MLIRContext &context, llvm::StringRef sourceName,
                            SourcePosition position) {
  return FileLineColLoc::get(&context, sourceName, position.line, position.column);
}

static OwningOpRef<ModuleOp> generateModule(const CheckedKernel &kernel, llvm::StringRef sourceName,
                                            MLIRContext &context) {
  context
      .loadDialect<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                   math::MathDialect, memref::MemRefDialect, tensor::TensorDialect, scf::SCFDialect,
                   cf::ControlFlowDialect, ir::OndrixDialect, ondsp::OndspDialect>();
  OpBuilder builder(&context);
  Location kernelLocation = getLocation(context, sourceName, kernel.ast.position);
  OwningOpRef<ModuleOp> module = ModuleOp::create(kernelLocation);

  auto getStorageType = [&](SourceType type) -> Type {
    if (type == SourceType::I32)
      return builder.getI32Type();
    if (type == SourceType::I64)
      return builder.getI64Type();
    if (type == SourceType::Bool)
      return builder.getI1Type();
    if (type == SourceType::Index)
      return builder.getIndexType();
    if (type == SourceType::Q15)
      return builder.getI16Type();
    if (type == SourceType::Q31 || type == SourceType::ComplexQ15)
      return builder.getI32Type();
    if (type == SourceType::ComplexQ31)
      return builder.getI64Type();
    if (type == SourceType::U8)
      return builder.getI8Type();
    return builder.getF32Type();
  };
  Type elementType = getStorageType(kernel.ast.primaryResult().type);
  // complex_f32 is the one declared type whose extent is not its element
  // count: each value occupies two adjacent elements of the trailing
  // dimension, which is what `interleaved` means.
  auto materializeShape = [](llvm::ArrayRef<std::optional<int64_t>> shape, SourceType type) {
    SmallVector<int64_t> dimensions;
    dimensions.reserve(shape.size());
    for (std::optional<int64_t> extent : shape)
      dimensions.push_back(extent.value_or(ShapedType::kDynamic));
    if (type == SourceType::ComplexF32 && !dimensions.empty() &&
        dimensions.back() != ShapedType::kDynamic)
      dimensions.back() *= 2;
    return dimensions;
  };
  SmallVector<Type> inputTypes;
  for (const ParameterAst &parameter : kernel.ast.parameters) {
    if (parameter.isConstexpr())
      continue;
    Type parameterElementType = getStorageType(parameter.type);
    if (parameter.isScalar())
      inputTypes.push_back(parameterElementType);
    else if (parameter.isTensor())
      inputTypes.push_back(RankedTensorType::get(materializeShape(parameter.shape, parameter.type),
                                                 parameterElementType));
    else
      inputTypes.push_back(
          MemRefType::get(materializeShape(parameter.shape, parameter.type), parameterElementType));
  }
  SmallVector<Type> resultTypes;
  resultTypes.reserve(kernel.ast.results.size());
  for (const ResultTypeAst &result : kernel.ast.results) {
    Type resultElementType = getStorageType(result.type);
    resultTypes.push_back(result.tensor
                              ? Type(RankedTensorType::get(
                                    materializeShape(result.shape, result.type), resultElementType))
                              : resultElementType);
  }
  Type resultType = resultTypes.front();
  FunctionType functionType = builder.getFunctionType(inputTypes, resultTypes);
  auto function = func::FuncOp::create(kernelLocation, kernel.ast.name, functionType);
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::DenseMap<llvm::StringRef, Value> arguments;
  unsigned argumentIndex = 0;
  for (const ParameterAst &parameter : kernel.ast.parameters) {
    if (!parameter.isConstexpr()) {
      // The source name travels as the argument's location, where the C entry
      // point reads it and nothing downstream has to carry or drop it.
      BlockArgument argument = entry->getArgument(argumentIndex++);
      argument.setLoc(NameLoc::get(builder.getStringAttr(parameter.name),
                                   getLocation(context, sourceName, parameter.position)));
      arguments.insert({parameter.name, argument});
      continue;
    }

    int64_t extent = static_cast<int64_t>(parameter.constantValues.size());
    Type parameterElementType = getStorageType(parameter.type);
    MemRefType coefficientType = MemRefType::get({extent}, parameterElementType);
    RankedTensorType initializerType = RankedTensorType::get({extent}, parameterElementType);
    SmallVector<llvm::APInt> values;
    values.reserve(parameter.constantValues.size());
    unsigned storageWidth = cast<IntegerType>(parameterElementType).getWidth();
    for (int64_t value : parameter.constantValues)
      values.emplace_back(storageWidth, static_cast<uint64_t>(value), true);
    auto initializer = DenseIntElementsAttr::get(initializerType, values);
    std::string symbolName = "__ox_" + kernel.ast.name + "_" + parameter.name;

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module->getBody());
    builder.create<memref::GlobalOp>(getLocation(context, sourceName, parameter.position),
                                     symbolName, builder.getStringAttr("private"), coefficientType,
                                     initializer, true, IntegerAttr());
    builder.setInsertionPointToStart(entry);
    Value coefficients = builder.create<memref::GetGlobalOp>(
        getLocation(context, sourceName, parameter.position), coefficientType, symbolName);
    if (kernel.ast.result.kind == ReductionKind::FirFilter)
      coefficients = builder.create<bufferization::ToTensorOp>(
          getLocation(context, sourceName, parameter.position), initializerType, coefficients,
          /*restrict=*/true, /*writable=*/false);
    arguments.insert({parameter.name, coefficients});
  }

  Location expressionLocation = getLocation(context, sourceName, kernel.ast.result.position);
  if (isComposableKernel(kernel.ast.result.kind, kernel.ast.primaryResult().type)) {
    auto layout = ondsp::CxLayoutAttr::get(&context, ondsp::ComplexLayout::PackedI16ImagHiRealLo);
    auto i16 = builder.getI16Type();
    auto numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, i16, 15);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto productScale = ondsp::ScaleAttr::get(&context, 0, 15, ondsp::RoundingMode::NearestEven,
                                              ondsp::OverflowMode::Saturate, i16);
    auto outputScale = ondsp::ScaleAttr::get(&context, 0, 1, ondsp::RoundingMode::NearestEven,
                                             ondsp::OverflowMode::Saturate, i16);
    // The re-frozen packed-Q31 profile shared by the i64-element FFT
    // builtins; sema admitted no other stage policy.
    auto i32 = builder.getI32Type();
    auto q31Layout =
        ondsp::CxLayoutAttr::get(&context, ondsp::ComplexLayout::PackedI32ImagHiRealLo);
    auto q31Numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, i32, 31);
    auto q31Product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::HighRaw);
    auto q31StageProduct = ondsp::ScaleAttr::get(
        &context, 1, 0, ondsp::RoundingMode::TowardNegative, ondsp::OverflowMode::Saturate, i32);
    auto q31StageOutput = ondsp::ScaleAttr::get(&context, 0, 1, ondsp::RoundingMode::TowardNegative,
                                                ondsp::OverflowMode::Saturate, i32);
    // The interleaved floating-point profile: sema admitted the contract mode
    // per call site and refused every requantization parameter.
    auto interleavedLayout = ondsp::CxLayoutAttr::get(&context, ondsp::ComplexLayout::Interleaved);
    auto getFpNumeric = [&](const BuiltinCallAst &call) {
      return ondsp::FpAttr::get(&context, builder.getF32Type(), *parseFpContract(call.fpContract));
    };
    std::function<Value(const BuiltinCallAst &)> emitComposedCall =
        [&](const BuiltinCallAst &call) -> Value {
      if (call.library) {
        // The checked kernel's body, cloned with its arguments bound here.
        OwningOpRef<ModuleOp> library = generateModule(*call.library, sourceName, context);
        Block &body = (*library->getOps<func::FuncOp>().begin()).getBody().front();
        IRMapping mapping;
        for (auto [argument, operand] : llvm::zip_equal(body.getArguments(), call.operands))
          mapping.map(argument, arguments.lookup(operand.parameter));
        for (Operation &op : body.without_terminator())
          builder.clone(op, mapping);
        return mapping.lookup(body.getTerminator()->getOperand(0));
      }
      if (call.kind == ReductionKind::Primitive) {
        Location loc = getLocation(context, sourceName, call.position);
        auto emitOperand = [&](const ExpressionAst &operand) -> Value {
          return operand.isParameterReference() ? arguments.lookup(operand.parameter)
                                                : emitComposedCall(*operand.call);
        };
        auto op = call.primitive;
        Type type = getStorageType(call.expressionType);
        if (op == PrimitiveKind::Literal) {
          if (type.isF32())
            return builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(call.fpConstant));
          return builder.create<arith::ConstantOp>(
              loc, builder.getIntegerAttr(type, *call.integerConstant));
        }
        auto lengthOf = [&](Value base) -> Value {
          if (isa<MemRefType>(base.getType()))
            return builder.create<memref::DimOp>(loc, base, 0);
          return builder.create<tensor::DimOp>(loc, base, 0);
        };
        std::function<void(const BuiltinCallAst &, size_t)> emitStatements;
        auto emitLoop = [&](const BuiltinCallAst &loop) {
          Location loopLoc = getLocation(context, sourceName, loop.position);
          Value start = emitOperand(loop.operands[0]);
          Value stop = emitOperand(loop.operands[1]);
          Type type = start.getType();
          unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
          auto constantOf = [&](const APInt &value) -> Value {
            return builder.create<arith::ConstantOp>(loopLoc, builder.getIntegerAttr(type, value));
          };
          Value zero = constantOf(APInt(width, 0));
          Value one = constantOf(APInt(width, 1));
          SmallVector<Value> initial;
          for (const std::string &name : loop.carriedNames)
            initial.push_back(arguments.lookup(name));
          auto emitBody = [&](Value induction, ValueRange carried) {
            auto saved = arguments;
            arguments[loop.bindingNames[0]] = induction;
            for (auto [name, value] : llvm::zip(loop.carriedNames, carried))
              arguments[name] = value;
            emitStatements(loop, 2);
            SmallVector<Value> next;
            for (const std::string &name : loop.carriedNames)
              next.push_back(arguments.lookup(name));
            arguments = std::move(saved);
            return next;
          };
          SmallVector<Value> results;
          if (loop.step == 1) {
            auto forOp = builder.create<scf::ForOp>(
                loopLoc, start, stop, one, initial,
                [&](OpBuilder &, Location, Value induction, ValueRange carried) {
                  builder.create<scf::YieldOp>(loopLoc, emitBody(induction, carried));
                });
            llvm::append_range(results, forOp.getResults());
          } else {
            // Count ceil(distance / |step|) exactly in unsigned arithmetic, so
            // no induction value is ever computed past the range.
            bool ascending = loop.step > 0;
            Value nonempty = builder.create<arith::CmpIOp>(
                loopLoc, ascending ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sgt, start,
                stop);
            Value distance = ascending ? builder.create<arith::SubIOp>(loopLoc, stop, start)
                                       : builder.create<arith::SubIOp>(loopLoc, start, stop);
            Value magnitude = constantOf(APInt(width, loop.step, true).abs());
            Value whole = builder.create<arith::DivUIOp>(
                loopLoc, builder.create<arith::SubIOp>(loopLoc, distance, one), magnitude);
            Value trips = builder.create<arith::SelectOp>(
                loopLoc, nonempty, builder.create<arith::AddIOp>(loopLoc, whole, one), zero);
            Value step = constantOf(APInt(width, loop.step, true));
            SmallVector<Value> operands{zero};
            llvm::append_range(operands, initial);
            auto whileOp = builder.create<scf::WhileOp>(
                loopLoc, ValueRange(operands).getTypes(), operands,
                [&](OpBuilder &, Location, ValueRange args) {
                  Value more = builder.create<arith::CmpIOp>(loopLoc, arith::CmpIPredicate::ne,
                                                             args.front(), trips);
                  builder.create<scf::ConditionOp>(loopLoc, more, args);
                },
                [&](OpBuilder &, Location, ValueRange args) {
                  Value offset = builder.create<arith::MulIOp>(loopLoc, args.front(), step);
                  Value induction = builder.create<arith::AddIOp>(loopLoc, start, offset);
                  SmallVector<Value> next{
                      builder.create<arith::AddIOp>(loopLoc, args.front(), one)};
                  llvm::append_range(next, emitBody(induction, args.drop_front()));
                  builder.create<scf::YieldOp>(loopLoc, next);
                });
            llvm::append_range(results, whileOp.getResults().drop_front());
          }
          for (auto [name, value] : llvm::zip(loop.carriedNames, results))
            arguments[name] = value;
        };
        emitStatements = [&](const BuiltinCallAst &owner, size_t begin) {
          for (size_t i = begin; i < owner.bindingNames.size(); ++i) {
            if (owner.bindingNames[i].empty())
              emitLoop(*owner.operands[i].call);
            else
              arguments[owner.bindingNames[i]] = emitOperand(owner.operands[i]);
          }
        };
        if (op == PrimitiveKind::Scope) {
          auto saved = arguments;
          emitStatements(call, 0);
          Value result = emitOperand(call.operands.back());
          arguments = std::move(saved);
          return result;
        }
        if (op == PrimitiveKind::Length)
          return lengthOf(emitOperand(call.operands.front()));
        if (op == PrimitiveKind::Element) {
          Value base = emitOperand(call.operands.front());
          Value index = emitOperand(call.operands[1]);
          Value position =
              index.getType().isIndex()
                  ? index
                  : builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), index);
          if (!call.inBounds) {
            // An i64 index is compared before narrowing to a 32-bit target index.
            Value length = lengthOf(base);
            Value inBounds =
                index.getType().isInteger(64)
                    ? builder.create<arith::CmpIOp>(
                          loc, arith::CmpIPredicate::ult, index,
                          builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), length))
                    : builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, position,
                                                    length);
            builder.create<cf::AssertOp>(loc, inBounds, "index out of bounds");
          }
          if (isa<MemRefType>(base.getType()))
            return builder.create<memref::LoadOp>(loc, base, position);
          return builder.create<tensor::ExtractOp>(loc, base, position);
        }
        Value lhs = emitOperand(call.operands.front());
        if (op == PrimitiveKind::Identity || op == PrimitiveKind::Pos || op == PrimitiveKind::Typed)
          return lhs;
        if (op == PrimitiveKind::And || op == PrimitiveKind::Or) {
          auto branch = builder.create<scf::IfOp>(loc, TypeRange{builder.getI1Type()}, lhs, true);
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&branch.getThenRegion().front());
          Value thenValue = op == PrimitiveKind::And
                                ? emitOperand(call.operands[1])
                                : builder.create<arith::ConstantIntOp>(loc, 1, 1);
          builder.create<scf::YieldOp>(loc, thenValue);
          builder.setInsertionPointToStart(&branch.getElseRegion().front());
          Value elseValue = op == PrimitiveKind::Or
                                ? emitOperand(call.operands[1])
                                : builder.create<arith::ConstantIntOp>(loc, 0, 1);
          builder.create<scf::YieldOp>(loc, elseValue);
          return branch.getResult(0);
        }
        Value rhs = call.operands.size() == 2 ? emitOperand(call.operands[1]) : Value();
        auto scalar = [&](OpBuilder &b, Value a, Value c) -> Value {
          bool fp = a.getType().isF32();
          if (op == PrimitiveKind::Not) {
            Value one = b.create<arith::ConstantIntOp>(loc, 1, 1);
            return b.create<arith::XOrIOp>(loc, a, one);
          }
          if (isComparison(op)) {
            if (fp) {
              auto pred = op == PrimitiveKind::Eq   ? arith::CmpFPredicate::OEQ
                          : op == PrimitiveKind::Ne ? arith::CmpFPredicate::UNE
                          : op == PrimitiveKind::Lt ? arith::CmpFPredicate::OLT
                          : op == PrimitiveKind::Le ? arith::CmpFPredicate::OLE
                          : op == PrimitiveKind::Gt ? arith::CmpFPredicate::OGT
                                                    : arith::CmpFPredicate::OGE;
              return b.create<arith::CmpFOp>(loc, pred, a, c);
            }
            auto pred = op == PrimitiveKind::Eq   ? arith::CmpIPredicate::eq
                        : op == PrimitiveKind::Ne ? arith::CmpIPredicate::ne
                        : op == PrimitiveKind::Lt ? arith::CmpIPredicate::slt
                        : op == PrimitiveKind::Le ? arith::CmpIPredicate::sle
                        : op == PrimitiveKind::Gt ? arith::CmpIPredicate::sgt
                                                  : arith::CmpIPredicate::sge;
            return b.create<arith::CmpIOp>(loc, pred, a, c);
          }
          if (fp) {
            switch (op) {
            case PrimitiveKind::Neg:
              return b.create<arith::NegFOp>(loc, a);
            case PrimitiveKind::Add:
              return b.create<arith::AddFOp>(loc, a, c);
            case PrimitiveKind::Sub:
              return b.create<arith::SubFOp>(loc, a, c);
            case PrimitiveKind::Mul:
              return b.create<arith::MulFOp>(loc, a, c);
            case PrimitiveKind::Div:
              return b.create<arith::DivFOp>(loc, a, c);
            default:
              llvm_unreachable("checked floating expression");
            }
          }
          if (a.getType().isIndex()) {
            // Index +/- traps on signed overflow of the target's index width.
            Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
            Value lhs = op == PrimitiveKind::Neg ? zero : a;
            Value rhs = op == PrimitiveKind::Neg ? a : c;
            bool add = op == PrimitiveKind::Add;
            Value result = add ? b.create<arith::AddIOp>(loc, lhs, rhs).getResult()
                               : b.create<arith::SubIOp>(loc, lhs, rhs).getResult();
            auto differ = [&](Value x, Value y) { return b.create<arith::XOrIOp>(loc, x, y); };
            Value overflow =
                add ? b.create<arith::AndIOp>(loc, differ(lhs, result), differ(rhs, result))
                    : b.create<arith::AndIOp>(loc, differ(lhs, rhs), differ(lhs, result));
            b.create<cf::AssertOp>(
                loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, overflow, zero),
                "index arithmetic overflow");
            return result;
          }
          if (op == PrimitiveKind::Neg) {
            Value zero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(a.getType(), 0));
            return b.create<arith::SubIOp>(loc, zero, a);
          }
          if (op == PrimitiveKind::Add)
            return b.create<arith::AddIOp>(loc, a, c);
          if (op == PrimitiveKind::Sub)
            return b.create<arith::SubIOp>(loc, a, c);
          if (op == PrimitiveKind::Mul)
            return b.create<arith::MulIOp>(loc, a, c);
          unsigned width = cast<IntegerType>(a.getType()).getWidth();
          auto constant = [&](APInt value) -> Value {
            return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(a.getType(), value));
          };
          Value zero = constant(APInt(width, 0));
          Value minimum = constant(APInt::getSignedMinValue(width));
          Value minusOne = constant(APInt::getAllOnes(width));
          Value nonzero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, c, zero);
          Value isMin = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, a, minimum);
          Value isMinusOne = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, c, minusOne);
          Value overflow = b.create<arith::AndIOp>(loc, isMin, isMinusOne);
          Value one = b.create<arith::ConstantIntOp>(loc, 1, 1);
          Value safe =
              b.create<arith::AndIOp>(loc, nonzero, b.create<arith::XOrIOp>(loc, overflow, one));
          b.create<cf::AssertOp>(loc, safe, "integer division by zero or overflow");
          return b.create<arith::DivSIOp>(loc, a, c);
        };
        if (!call.expressionExtent)
          return scalar(builder, lhs, rhs);
        auto tensorType = RankedTensorType::get({call.expressionExtent}, type);
        Value init = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(), type);
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value end = builder.create<arith::ConstantIndexOp>(loc, call.expressionExtent);
        Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
        auto loop = builder.create<scf::ForOp>(
            loc, zero, end, one, ValueRange{init},
            [&](OpBuilder &b, Location loc, Value i, ValueRange args) {
              auto element = [&](Value value) -> Value {
                if (value && isa<RankedTensorType>(value.getType()))
                  return b.create<tensor::ExtractOp>(loc, value, ValueRange{i});
                return value;
              };
              Value result = scalar(b, element(lhs), element(rhs));
              Value out = b.create<tensor::InsertOp>(loc, result, args.front(), ValueRange{i});
              b.create<scf::YieldOp>(loc, out);
            });
        return loop.getResult(0);
      }
      if (call.kind == ReductionKind::FirFilter) {
        // The composed filter stage: sema pinned one uniform width, the valid
        // boundary, static extents, and that width's exact accumulator profile.
        Location callLocation = getLocation(context, sourceName, call.position);
        Value signal = arguments.lookup(call.operands[0].parameter);
        auto signalType = cast<RankedTensorType>(signal.getType());
        auto storage = cast<IntegerType>(signalType.getElementType());
        ondsp::FixedAttr stageNumeric = storage.getWidth() == 32 ? q31Numeric : numeric;
        const ExpressionAst &coefficientOperand = call.operands[1];
        Value coefficients;
        if (coefficientOperand.isParameterReference()) {
          coefficients = arguments.lookup(coefficientOperand.parameter);
        } else {
          const BuiltinCallAst &design = *coefficientOperand.call;
          Location designLocation = getLocation(context, sourceName, design.position);
          auto designType = RankedTensorType::get({design.taps}, storage);
          switch (design.kind) {
          case ReductionKind::Hamming:
            coefficients =
                builder.create<ir::WindowHammingOp>(designLocation, designType, stageNumeric);
            break;
          case ReductionKind::Hann:
            coefficients =
                builder.create<ir::WindowHannOp>(designLocation, designType, stageNumeric);
            break;
          case ReductionKind::Blackman:
            coefficients =
                builder.create<ir::WindowBlackmanOp>(designLocation, designType, stageNumeric);
            break;
          case ReductionKind::Kaiser:
            coefficients = builder.create<ir::WindowKaiserOp>(
                designLocation, designType, builder.getI64IntegerAttr(design.betaNum),
                builder.getI64IntegerAttr(design.betaDen), stageNumeric);
            break;
          default:
            coefficients = builder.create<ir::FirDesignWindowedSincOp>(
                designLocation, designType,
                ir::FirDesignResponseAttr::get(&context, ir::FirDesignResponse::Lowpass),
                builder.getI64IntegerAttr(design.cutoffNum),
                builder.getI64IntegerAttr(design.cutoffDen), stageNumeric);
            break;
          }
        }
        int64_t inputExtent = signalType.getDimSize(0);
        int64_t tapCount = cast<RankedTensorType>(coefficients.getType()).getDimSize(0);
        auto outputType = RankedTensorType::get({inputExtent - tapCount + 1}, storage);
        Value init = builder.create<tensor::EmptyOp>(callLocation, outputType.getShape(), storage);
        auto accumulatorType = ondsp::AccType::get(
            &context, builder.getIntegerType(call.accumulatorWidth), 2 * stageNumeric.getFrac(),
            ondsp::Signedness::Signed, *parseOverflow(call.updateOverflow));
        return builder.create<ir::FirFilterOp>(
            callLocation, outputType, signal, coefficients, init, Value(),
            ir::FirBoundaryMode::Valid, stageNumeric, product, TypeAttr::get(accumulatorType),
            stageNumeric, ondsp::RoundingModeAttr::get(&context, *parseRounding(call.rounding)),
            ondsp::OverflowModeAttr::get(&context, *parseOverflow(call.destinationOverflow)));
      }
      const ExpressionAst &operand = call.operands.front();
      Value input = operand.isParameterReference() ? arguments.lookup(operand.parameter)
                                                   : emitComposedCall(*operand.call);
      auto inputType = cast<RankedTensorType>(input.getType());
      if (call.kind == ReductionKind::ViterbiDecode) {
        int64_t rate = static_cast<int64_t>(call.polynomials.size());
        auto bitsType =
            RankedTensorType::get({inputType.getDimSize(0) / rate / 8}, builder.getI8Type());
        return builder.create<ir::ViterbiDecodeOp>(
            getLocation(context, sourceName, call.position), bitsType, input,
            builder.getI64IntegerAttr(call.constraintLength),
            builder.getDenseI64ArrayAttr(call.polynomials),
            call.symbolBound ? builder.getI64IntegerAttr(call.symbolBound) : IntegerAttr());
      }
      if (isConversionKind(call.kind)) {
        // Sema pinned the direction, so the operand's element type and the
        // spelled target name the two formats.
        auto formatOf = [&](Type element) -> Attribute {
          if (element.isF32())
            return ondsp::FpAttr::get(&context, element, ondsp::FpContractMode::Off);
          return element.isSignlessInteger(32) ? q31Numeric : numeric;
        };
        Type target = call.target == SourceType::F32   ? Type(builder.getF32Type())
                      : call.target == SourceType::Q31 ? Type(i32)
                                                       : Type(i16);
        bool lossy = isLossyConversionKind(call.kind);
        auto resultType = RankedTensorType::get({inputType.getDimSize(0)}, target);
        return builder.create<ir::QuantizeOp>(
            getLocation(context, sourceName, call.position), resultType, input,
            formatOf(inputType.getElementType()), formatOf(target),
            lossy ? ondsp::RoundingModeAttr::get(&context, *parseRounding(call.rounding))
                  : ondsp::RoundingModeAttr(),
            lossy ? ondsp::OverflowModeAttr::get(&context, *parseOverflow(call.destinationOverflow))
                  : ondsp::OverflowModeAttr());
      }
      if (call.kind == ReductionKind::Gain) {
        ondsp::FixedAttr gainNumeric =
            inputType.getElementType().isSignlessInteger(32) ? q31Numeric : numeric;
        return builder.create<ir::GainOp>(
            getLocation(context, sourceName, call.position), inputType, input,
            builder.getI64IntegerAttr(call.gain), FloatAttr(), Attribute(gainNumeric),
            ondsp::RoundingModeAttr::get(&context, *parseRounding(call.rounding)));
      }
      if (isElementwiseKind(call.kind)) {
        Location callLocation = getLocation(context, sourceName, call.position);
        auto rounding = ondsp::RoundingModeAttr::get(&context, *parseRounding(call.rounding));
        auto overflow =
            ondsp::OverflowModeAttr::get(&context, *parseOverflow(call.destinationOverflow));
        // Sema proved every operand carries the same declared width, so the
        // operand storage is what selects the profile.
        ondsp::FixedAttr elementwise =
            inputType.getElementType().isSignlessInteger(32) ? q31Numeric : numeric;
        if (isBinaryElementwiseKind(call.kind)) {
          const ExpressionAst &second = call.operands[1];
          Value rhs = second.isParameterReference() ? arguments.lookup(second.parameter)
                                                    : emitComposedCall(*second.call);
          if (call.kind == ReductionKind::Ratio) {
            auto policy = call.nonpositive == "saturate" ? ondsp::NonpositiveDivisor::Saturate
                                                         : ondsp::NonpositiveDivisor::Trap;
            return builder.create<ir::RatioOp>(
                callLocation, inputType, input, rhs, elementwise, rounding, overflow,
                ondsp::NonpositiveDivisorAttr::get(&context, policy));
          }
          if (call.kind == ReductionKind::Add)
            return builder.create<ir::AddOp>(callLocation, inputType, input, rhs, elementwise,
                                             overflow);
          if (call.kind == ReductionKind::Sub)
            return builder.create<ir::SubOp>(callLocation, inputType, input, rhs, elementwise,
                                             overflow);
          return builder.create<ir::MultOp>(callLocation, inputType, input, rhs, elementwise,
                                            rounding, overflow);
        }
        if (call.kind == ReductionKind::Abs)
          return builder.create<ir::AbsOp>(callLocation, inputType, input, elementwise, overflow);
        if (call.kind == ReductionKind::Negate)
          return builder.create<ir::NegateOp>(callLocation, inputType, input, elementwise,
                                              overflow);
        if (call.kind == ReductionKind::Offset)
          return builder.create<ir::OffsetOp>(callLocation, inputType, input,
                                              builder.getI64IntegerAttr(call.bias), elementwise,
                                              overflow);
        if (call.kind == ReductionKind::Div)
          return builder.create<ir::DivOp>(callLocation, inputType, input,
                                           builder.getI64IntegerAttr(call.divisor), elementwise,
                                           rounding, overflow);
        return builder.create<ir::ShiftOp>(callLocation, inputType, input,
                                           builder.getI64IntegerAttr(call.amount), elementwise,
                                           rounding, overflow);
      }
      Location callLocation = getLocation(context, sourceName, call.position);
      if (call.kind == ReductionKind::Phase) {
        if (inputType.getElementType().isF32()) {
          auto outputType =
              RankedTensorType::get({inputType.getDimSize(0) / 2}, builder.getF32Type());
          return builder.create<ir::CxPhaseOp>(callLocation, outputType, input, interleavedLayout,
                                               getFpNumeric(call), ondsp::FixedAttr(),
                                               ondsp::RoundingModeAttr());
        }
        // The result reading is the unsigned turn at the declared width; the
        // source type system names only the storage, so the binding supplies
        // the reading — the same projection log2/exp2 use.
        bool wideComponents = inputType.getElementType().isSignlessInteger(64);
        IntegerType turnStorage = call.turn == "q31" ? i32 : i16;
        auto outputType = RankedTensorType::get({inputType.getDimSize(0)}, turnStorage);
        auto turn = ondsp::FixedAttr::get(&context, ondsp::Signedness::Unsigned, turnStorage,
                                          turnStorage.getWidth());
        return builder.create<ir::CxPhaseOp>(
            callLocation, outputType, input, wideComponents ? q31Layout : layout,
            wideComponents ? q31Numeric : numeric, turn,
            ondsp::RoundingModeAttr::get(&context, ondsp::RoundingMode::NearestEven));
      }
      if (call.kind == ReductionKind::Power) {
        // One interleaved bin is two elements, so the readout is half as long
        // as the operand and carries no declared boundary at all.
        if (inputType.getElementType().isF32()) {
          auto outputType =
              RankedTensorType::get({inputType.getDimSize(0) / 2}, builder.getF32Type());
          return builder.create<ir::CxPowerOp>(callLocation, outputType, input, interleavedLayout,
                                               getFpNumeric(call), ondsp::FixedAttr(),
                                               ondsp::RoundingModeAttr());
        }
        auto outputType =
            RankedTensorType::get({inputType.getDimSize(0)}, builder.getIntegerType(16));
        // The export default, which is also what the packed complex unit
        // rounds with; omission keeps it.
        ondsp::RoundingMode rounding = ondsp::RoundingMode::NearestTiesPositive;
        if (!call.rounding.empty())
          rounding = *parseRounding(call.rounding);
        return builder.create<ir::CxPowerOp>(
            callLocation, outputType, input, layout, numeric,
            ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, builder.getI16Type(), 15),
            ondsp::RoundingModeAttr::get(&context, rounding));
      }
      if (call.kind == ReductionKind::Magnitude) {
        if (inputType.getElementType().isF32()) {
          auto outputType =
              RankedTensorType::get({inputType.getDimSize(0) / 2}, builder.getF32Type());
          return builder.create<ir::CxMagnitudeOp>(
              callLocation, outputType, input, interleavedLayout, getFpNumeric(call),
              ondsp::RoundingModeAttr(), ondsp::RoundingModeAttr());
        }
        // The packed container the operand carries selects the profile, and
        // the Q31 sum of squares takes the pre-shift boundary its width
        // forces.
        bool packedQ31 = inputType.getElementType().isSignlessInteger(64);
        ondsp::CxLayoutAttr magnitudeLayout = packedQ31 ? q31Layout : layout;
        Attribute magnitudeNumeric = packedQ31 ? Attribute(q31Numeric) : Attribute(numeric);
        unsigned componentWidth =
            ondsp::getPackedComplexProfile(magnitudeLayout.getLayout())->storageWidth;
        auto outputType = RankedTensorType::get({inputType.getDimSize(0)},
                                                builder.getIntegerType(componentWidth));
        // Both roundings were validated in sema; omission keeps the
        // nearest_even default, and the pre-shift attribute exists exactly
        // where the width derives a pre-shift.
        ondsp::RoundingModeAttr inputRounding;
        if (ir::getRmsInputPreShift(componentWidth, /*extent=*/2) > 0)
          inputRounding = ondsp::RoundingModeAttr::get(
              &context, call.inputRounding.empty() ? ondsp::RoundingMode::NearestEven
                                                   : *parseRounding(call.inputRounding));
        ondsp::RoundingMode rootRounding = ondsp::RoundingMode::NearestEven;
        if (!call.rounding.empty())
          rootRounding = *parseRounding(call.rounding);
        return builder.create<ir::CxMagnitudeOp>(
            callLocation, outputType, input, magnitudeLayout, magnitudeNumeric, inputRounding,
            ondsp::RoundingModeAttr::get(&context, rootRounding));
      }
      if (isCfftKind(call.kind)) {
        auto direction = ir::CfftDirectionAttr::get(&context, call.kind == ReductionKind::Cfft
                                                                  ? ir::CfftDirection::Forward
                                                                  : ir::CfftDirection::Inverse);
        if (inputType.getElementType().isF32())
          return builder.create<ir::CfftOp>(
              callLocation, inputType, input, direction, interleavedLayout, getFpNumeric(call),
              ondsp::ProductAttr(), ondsp::ScaleAttr(), ondsp::ScaleAttr());
        if (inputType.getElementType().isInteger(64))
          return builder.create<ir::CfftOp>(callLocation, inputType, input, direction, q31Layout,
                                            q31Numeric, q31Product, q31StageProduct,
                                            q31StageOutput);
        ondsp::RoundingMode stageRounding = call.rounding.empty() ? ondsp::RoundingMode::NearestEven
                                                                  : *parseRounding(call.rounding);
        ondsp::OverflowMode stageOverflow = call.destinationOverflow.empty()
                                                ? ondsp::OverflowMode::Saturate
                                                : *parseOverflow(call.destinationOverflow);
        auto stageProduct = ondsp::ScaleAttr::get(&context, 0, 15, stageRounding, stageOverflow,
                                                  builder.getI16Type());
        auto stageOutput = ondsp::ScaleAttr::get(&context, 0, 1, stageRounding, stageOverflow,
                                                 builder.getI16Type());
        return builder.create<ir::CfftOp>(callLocation, inputType, input, direction, layout,
                                          numeric, product, stageProduct, stageOutput);
      }
      if (call.kind == ReductionKind::Rfft) {
        int64_t realExtent = inputType.getDimSize(0);
        if (inputType.getElementType().isF32()) {
          auto outputType = RankedTensorType::get({realExtent + 2}, builder.getF32Type());
          return builder.create<ir::RfftOp>(callLocation, outputType, input, interleavedLayout,
                                            getFpNumeric(call), ondsp::ProductAttr(),
                                            ondsp::ScaleAttr(), ondsp::ScaleAttr());
        }
        // Q31 real samples arrive as i32 elements and produce packed i64
        // bins under the frozen profile.
        if (inputType.getElementType().isInteger(32)) {
          auto outputType = RankedTensorType::get({realExtent / 2 + 1}, builder.getI64Type());
          return builder.create<ir::RfftOp>(callLocation, outputType, input, q31Layout, q31Numeric,
                                            q31Product, q31StageProduct, q31StageOutput);
        }
        auto outputType = RankedTensorType::get({realExtent / 2 + 1}, builder.getI32Type());
        return builder.create<ir::RfftOp>(callLocation, outputType, input, layout, numeric, product,
                                          productScale, outputScale);
      }
      if (inputType.getElementType().isF32()) {
        int64_t bins = inputType.getDimSize(0) / 2;
        auto outputType = RankedTensorType::get({(bins - 1) * 2}, builder.getF32Type());
        return builder.create<ir::IrfftOp>(callLocation, outputType, input, interleavedLayout,
                                           getFpNumeric(call), ondsp::ProductAttr(),
                                           ondsp::ScaleAttr(), ondsp::ScaleAttr());
      }
      int64_t realExtent = (inputType.getDimSize(0) - 1) * 2;
      if (inputType.getElementType().isInteger(64)) {
        auto outputType = RankedTensorType::get({realExtent}, builder.getI32Type());
        return builder.create<ir::IrfftOp>(callLocation, outputType, input, q31Layout, q31Numeric,
                                           q31Product, q31StageProduct, q31StageOutput);
      }
      auto outputType = RankedTensorType::get({realExtent}, builder.getI16Type());
      return builder.create<ir::IrfftOp>(callLocation, outputType, input, layout, numeric, product,
                                         productScale, outputScale);
    };
    Value result = emitComposedCall(kernel.ast.result);
    builder.create<func::ReturnOp>(expressionLocation, result);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }

  Value lhs = arguments.lookup(*getParameterOperand(kernel.ast.result, 0));
  if (kernel.ast.result.kind == ReductionKind::Rms && !isa<RankedTensorType>(resultType)) {
    // Only the reduction is contract indexed, so the scalar spelling rides
    // the dot route; the division and the root are single IEEE events.
    auto numeric = ondsp::FpAttr::get(&context, elementType, *kernel.fpContract);
    Value sumOfSquares = builder.create<ir::DotOp>(expressionLocation, elementType, lhs, lhs,
                                                   numeric, ondsp::ProductAttr(), IntegerAttr());
    int64_t extent = cast<MemRefType>(lhs.getType()).getDimSize(0);
    Value count = builder.create<arith::ConstantOp>(
        expressionLocation, builder.getF32FloatAttr(static_cast<float>(extent)));
    Value mean = builder.create<arith::DivFOp>(expressionLocation, sumOfSquares, count);
    Value root = builder.create<math::SqrtOp>(expressionLocation, mean);
    builder.create<func::ReturnOp>(expressionLocation, root);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (isUnaryTensorKind(kernel.ast.result.kind)) {
    auto outputType = cast<RankedTensorType>(resultType);
    ondsp::FixedAttr numeric;
    ondsp::RoundingModeAttr rounding;
    if (!kernel.fpContract) {
      numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType,
                                      cast<IntegerType>(elementType).getWidth() - 1);
      rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
    }
    Value result;
    if (kernel.ast.result.kind == ReductionKind::Dct) {
      if (kernel.fpContract) {
        auto fp = ondsp::FpAttr::get(&context, elementType, *kernel.fpContract);
        result = builder.create<ir::DctOp>(expressionLocation, outputType, lhs, fp, fp,
                                           ondsp::RoundingModeAttr(), ondsp::RoundingModeAttr(),
                                           ondsp::ProductAttr());
      } else {
        unsigned storageWidth = cast<IntegerType>(elementType).getWidth();
        unsigned stageCount = llvm::Log2_64(outputType.getDimSize(0));
        auto outputNumeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType,
                                                   storageWidth - 2 - stageCount);
        // Omission keeps the nearest_even default; only a departing
        // declaration is materialized.
        auto declared = *kernel.rounding != ondsp::RoundingMode::NearestEven
                            ? ondsp::RoundingModeAttr::get(&context, *kernel.rounding)
                            : ondsp::RoundingModeAttr();
        // Required exactly where the row sum would leave i64, which is every
        // Q31 extent and no Q15 one.
        bool rawHigh = kernel.ast.result.product == "raw_high";
        auto product =
            !rawHigh && ir::getReductionProductShift(storageWidth, outputType.getDimSize(0)) > 0
                ? ondsp::RoundingModeAttr::get(
                      &context, kernel.productRounding.value_or(ondsp::RoundingMode::NearestEven))
                : ondsp::RoundingModeAttr();
        auto selection = rawHigh
                             ? ondsp::ProductAttr::get(&context, ondsp::ProductSelection::HighRaw)
                             : ondsp::ProductAttr();
        result = builder.create<ir::DctOp>(expressionLocation, outputType, lhs, numeric,
                                           outputNumeric, product, declared, selection);
      }
    } else if (kernel.ast.result.kind == ReductionKind::MovingAverage) {
      result = builder.create<ir::MovingAverageOp>(
          expressionLocation, outputType, lhs, builder.getI64IntegerAttr(kernel.ast.result.window),
          kernel.fpContract
              ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
              : Attribute(numeric));
    } else if (kernel.ast.result.kind == ReductionKind::Gain) {
      result = builder.create<ir::GainOp>(
          expressionLocation, outputType, lhs,
          kernel.fpContract ? IntegerAttr() : builder.getI64IntegerAttr(kernel.ast.result.gain),
          kernel.fpContract ? builder.getF32FloatAttr(kernel.ast.result.fpConstant) : FloatAttr(),
          kernel.fpContract
              ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
              : Attribute(numeric),
          rounding);
    } else if (kernel.ast.result.kind == ReductionKind::CicDecimate) {
      result = builder.create<ir::CicDecimateOp>(
          expressionLocation, outputType, lhs, builder.getI64IntegerAttr(kernel.ast.result.stages),
          builder.getI64IntegerAttr(kernel.ast.result.rate),
          builder.getI64IntegerAttr(kernel.ast.result.delay), numeric,
          ondsp::OverflowModeAttr::get(&context, *kernel.stateOverflow), rounding);
    } else if (kernel.ast.result.kind == ReductionKind::Log2 ||
               kernel.ast.result.kind == ReductionKind::Exp2) {
      // The source type system names only the storage, so the two readings
      // the contract distinguishes are supplied here rather than spelled at
      // the call site; the projection is safe because the pair does not
      // compose with anything that would misread the scale.
      unsigned width = cast<IntegerType>(elementType).getWidth();
      auto magnitude =
          ondsp::FixedAttr::get(&context, ondsp::Signedness::Unsigned, elementType, width);
      auto exponent = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType,
                                            width == 16 ? 11 : 26);
      if (kernel.ast.result.kind == ReductionKind::Log2)
        result = builder.create<ir::Log2Op>(expressionLocation, outputType, lhs, magnitude,
                                            exponent, rounding);
      else
        result = builder.create<ir::Exp2Op>(expressionLocation, outputType, lhs, exponent,
                                            magnitude, rounding);
    } else if (kernel.ast.result.kind == ReductionKind::Goertzel) {
      // f32 goertzel rounds at no boundary of its own, so the optional
      // rounding attribute stays absent.
      result = builder.create<ir::GoertzelOp>(
          expressionLocation, outputType, lhs, builder.getI64IntegerAttr(kernel.ast.result.bin),
          ondsp::FpAttr::get(&context, elementType, *kernel.fpContract), ondsp::RoundingModeAttr(),
          ondsp::RoundingModeAttr());
    } else if (kernel.ast.result.kind == ReductionKind::Sine) {
      result = builder.create<ir::SineOp>(expressionLocation, outputType, lhs, numeric, rounding);
    } else if (kernel.ast.result.kind == ReductionKind::Cosine) {
      result = builder.create<ir::CosineOp>(expressionLocation, outputType, lhs, numeric, rounding);
    } else {
      // The pre-shift boundary exists only where an exact sum of squares would
      // not fit i64, so the binding declares its rounding exactly there.
      ondsp::RoundingModeAttr inputRounding =
          kernel.inputRounding ? ondsp::RoundingModeAttr::get(&context, *kernel.inputRounding)
                               : ondsp::RoundingModeAttr();
      result = builder.create<ir::RmsOp>(
          expressionLocation, outputType, lhs,
          kernel.fpContract
              ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
              : Attribute(numeric),
          inputRounding, kernel.fpContract ? ondsp::RoundingModeAttr() : rounding);
    }
    builder.create<func::ReturnOp>(expressionLocation, result);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (kernel.ast.result.kind == ReductionKind::Butterfly) {
    Value rhs = arguments.lookup(*getParameterOperand(kernel.ast.result, 1));
    Value twiddle = arguments.lookup(*getParameterOperand(kernel.ast.result, 2));
    auto layout = ondsp::CxLayoutAttr::get(&context, ondsp::ComplexLayout::PackedI16ImagHiRealLo);
    auto i16 = builder.getI16Type();
    auto numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, i16, 15);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto productScale = ondsp::ScaleAttr::get(&context, 0, 15, ondsp::RoundingMode::NearestEven,
                                              ondsp::OverflowMode::Saturate, i16);
    auto outputScale = ondsp::ScaleAttr::get(&context, 0, 1, ondsp::RoundingMode::NearestEven,
                                             ondsp::OverflowMode::Saturate, i16);
    auto butterfly = builder.create<ir::ButterflyOp>(expressionLocation, elementType, elementType,
                                                     lhs, rhs, twiddle, layout, numeric, product,
                                                     productScale, outputScale);
    builder.create<func::ReturnOp>(expressionLocation,
                                   ValueRange{butterfly.getOut0(), butterfly.getOut1()});
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }

  Value rhs = arguments.lookup(*getParameterOperand(kernel.ast.result, 1));
  if (kernel.ast.result.kind == ReductionKind::CxDot) {
    SourceType complexType = kernel.ast.primaryResult().type;
    unsigned componentWidth = getComplexComponentWidth(complexType);
    auto component = builder.getIntegerType(componentWidth);
    auto numeric =
        ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, component, componentWidth - 1);
    auto layout = ondsp::CxLayoutAttr::get(&context, getPackedComplexLayout(complexType));
    auto accumulatorType = ondsp::AccType::get(
        &context, builder.getIntegerType(kernel.ast.result.accumulatorWidth),
        2 * (componentWidth - 1), ondsp::Signedness::Signed, *kernel.updateOverflow);
    auto reduction = builder.create<ir::CxDotOp>(
        expressionLocation, accumulatorType, accumulatorType, lhs, rhs, numeric, layout,
        kernel.ast.result.conjugate ? builder.getUnitAttr() : UnitAttr());
    SmallVector<Value> components;
    for (Value accumulator : {reduction.getResultReal(), reduction.getResultImag()})
      components.push_back(
          builder.create<ondsp::AccExportOp>(expressionLocation, component, accumulator, numeric,
                                             *kernel.rounding, *kernel.destinationOverflow));
    // Zero extension, not sign extension: a negative real component would
    // otherwise flood the half the imaginary component occupies.
    Value real = builder.create<arith::ExtUIOp>(expressionLocation, elementType, components[0]);
    Value imaginary =
        builder.create<arith::ExtUIOp>(expressionLocation, elementType, components[1]);
    Value shift =
        builder.create<arith::ConstantIntOp>(expressionLocation, componentWidth, elementType);
    Value high = builder.create<arith::ShLIOp>(expressionLocation, imaginary, shift);
    Value packed = builder.create<arith::OrIOp>(expressionLocation, real, high);
    builder.create<func::ReturnOp>(expressionLocation, packed);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (kernel.ast.result.kind == ReductionKind::CxFirFilter) {
    auto outputType = cast<RankedTensorType>(resultType);
    SourceType complexType = kernel.ast.primaryResult().type;
    if (complexType == SourceType::ComplexF32) {
      Value init =
          builder.create<tensor::EmptyOp>(expressionLocation, outputType.getShape(), elementType);
      Value result = builder.create<ir::CxFirFilterOp>(
          expressionLocation, outputType, lhs, rhs, init,
          ondsp::FpAttr::get(&context, builder.getF32Type(), *kernel.fpContract),
          ondsp::CxLayoutAttr::get(&context, ondsp::ComplexLayout::Interleaved), TypeAttr(),
          ondsp::RoundingModeAttr(), ondsp::OverflowModeAttr(),
          kernel.ast.result.conjugate ? builder.getUnitAttr() : UnitAttr());
      builder.create<func::ReturnOp>(expressionLocation, result);
      module->push_back(function);
      if (failed(verify(*module)))
        return {};
      return module;
    }
    unsigned componentWidth = getComplexComponentWidth(complexType);
    auto component = builder.getIntegerType(componentWidth);
    auto numeric =
        ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, component, componentWidth - 1);
    auto layout = ondsp::CxLayoutAttr::get(&context, getPackedComplexLayout(complexType));
    auto accumulatorType = ondsp::AccType::get(
        &context, builder.getIntegerType(kernel.ast.result.accumulatorWidth),
        2 * (componentWidth - 1), ondsp::Signedness::Signed, *kernel.updateOverflow);
    Value init =
        builder.create<tensor::EmptyOp>(expressionLocation, outputType.getShape(), elementType);
    Value result = builder.create<ir::CxFirFilterOp>(
        expressionLocation, outputType, lhs, rhs, init, numeric, layout,
        TypeAttr::get(accumulatorType), ondsp::RoundingModeAttr::get(&context, *kernel.rounding),
        ondsp::OverflowModeAttr::get(&context, *kernel.destinationOverflow),
        kernel.ast.result.conjugate ? builder.getUnitAttr() : UnitAttr());
    builder.create<func::ReturnOp>(expressionLocation, result);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (kernel.ast.result.kind == ReductionKind::SosDf2Fixed) {
    Value scales = arguments.lookup(*getParameterOperand(kernel.ast.result, 2));
    Value state = arguments.lookup(*getParameterOperand(kernel.ast.result, 3));
    unsigned fractionalBits = cast<IntegerType>(elementType).getWidth() - 1;
    auto numeric =
        ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, fractionalBits);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto accumulatorType =
        ondsp::AccType::get(&context, builder.getIntegerType(kernel.ast.result.accumulatorWidth),
                            fractionalBits * 2, ondsp::Signedness::Signed, *kernel.updateOverflow);
    auto sos = builder.create<ir::SosFilterDf2FixedOp>(
        expressionLocation, resultTypes, lhs, rhs, scales, state, numeric, product, accumulatorType,
        *kernel.stateRounding, *kernel.stateOverflow, *kernel.rounding,
        *kernel.destinationOverflow);
    builder.create<func::ReturnOp>(expressionLocation,
                                   ValueRange{sos.getOutput(), sos.getNextState()});
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (kernel.ast.result.kind == ReductionKind::SosTdf2) {
    Value scales = arguments.lookup(*getParameterOperand(kernel.ast.result, 2));
    Value state = arguments.lookup(*getParameterOperand(kernel.ast.result, 3));
    auto sos = builder.create<ir::SosFilterTdf2Op>(
        expressionLocation, resultTypes, lhs, rhs, scales, state,
        ondsp::FpAttr::get(&context, elementType, *kernel.fpContract));
    builder.create<func::ReturnOp>(expressionLocation,
                                   ValueRange{sos.getOutput(), sos.getNextState()});
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (kernel.ast.result.kind == ReductionKind::FirStream) {
    Value state = arguments.lookup(*getParameterOperand(kernel.ast.result, 2));
    auto outputType = cast<RankedTensorType>(resultTypes[0]);
    auto nextStateType = cast<RankedTensorType>(resultTypes[1]);
    unsigned fractionalBits = cast<IntegerType>(elementType).getWidth() - 1;
    auto fixed =
        ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, fractionalBits);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto accumulatorType =
        ondsp::AccType::get(&context, builder.getIntegerType(kernel.ast.result.accumulatorWidth),
                            fractionalBits * 2, ondsp::Signedness::Signed, *kernel.updateOverflow);
    auto stream = builder.create<ir::FirStreamOp>(
        expressionLocation, TypeRange{outputType, nextStateType}, lhs, rhs, state, fixed, product,
        TypeAttr::get(accumulatorType), fixed,
        ondsp::RoundingModeAttr::get(&context, *kernel.rounding),
        ondsp::OverflowModeAttr::get(&context, *kernel.destinationOverflow));
    builder.create<func::ReturnOp>(expressionLocation,
                                   ValueRange{stream.getOutput(), stream.getNextState()});
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }

  if (kernel.ast.result.kind == ReductionKind::Matmul) {
    Attribute numeric =
        kernel.fpContract
            ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
            : Attribute(ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType,
                                              cast<IntegerType>(elementType).getWidth() - 1));
    ondsp::RoundingModeAttr rounding;
    if (!kernel.fpContract)
      rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
    ondsp::RoundingModeAttr productRounding =
        kernel.productRounding ? ondsp::RoundingModeAttr::get(&context, *kernel.productRounding)
                               : ondsp::RoundingModeAttr();
    ondsp::ProductAttr selection =
        kernel.ast.result.product == "raw_high"
            ? ondsp::ProductAttr::get(&context, ondsp::ProductSelection::HighRaw)
            : ondsp::ProductAttr();
    auto product =
        builder.create<ir::MatmulOp>(expressionLocation, cast<RankedTensorType>(resultType), lhs,
                                     rhs, numeric, productRounding, rounding, selection);
    builder.create<func::ReturnOp>(expressionLocation, product.getResult());
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (kernel.ast.result.kind == ReductionKind::Lms) {
    Value weights = arguments.lookup(*getParameterOperand(kernel.ast.result, 2));
    Attribute numeric =
        kernel.fpContract
            ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
            : Attribute(ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType,
                                              cast<IntegerType>(elementType).getWidth() - 1));
    ondsp::RoundingModeAttr rounding;
    if (!kernel.fpContract)
      rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
    ondsp::RoundingModeAttr productRounding =
        kernel.productRounding ? ondsp::RoundingModeAttr::get(&context, *kernel.productRounding)
                               : ondsp::RoundingModeAttr();
    auto lms = builder.create<ir::LmsOp>(
        expressionLocation, resultTypes[0], resultTypes[1], lhs, rhs, weights,
        kernel.fpContract ? IntegerAttr() : builder.getI64IntegerAttr(kernel.ast.result.stepSize),
        kernel.fpContract ? builder.getF32FloatAttr(kernel.ast.result.fpConstant) : FloatAttr(),
        productRounding, numeric, rounding,
        kernel.ast.result.normalized ? builder.getI64IntegerAttr(kernel.ast.result.epsilon)
                                     : IntegerAttr());
    builder.create<func::ReturnOp>(expressionLocation,
                                   ValueRange{lms.getError(), lms.getAdapted()});
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }

  bool isConv1D = kernel.ast.result.kind == ReductionKind::Convolution ||
                  kernel.ast.result.kind == ReductionKind::Correlation;
  bool isFirDecimate = kernel.ast.result.kind == ReductionKind::FirDecimate;
  bool isFirInterpolate = kernel.ast.result.kind == ReductionKind::FirInterpolate;
  if (kernel.ast.result.kind == ReductionKind::FirFilter || isFirDecimate || isFirInterpolate ||
      isConv1D) {
    auto outputType = cast<RankedTensorType>(resultType);
    // Resampling result extents are required to be static during checking.
    assert(!((isFirDecimate || isFirInterpolate) && outputType.isDynamicDim(0)) &&
           "resampling source result must be static");
    SmallVector<Value> dynamicSizes;
    if (outputType.isDynamicDim(0)) {
      Value zero = builder.create<arith::ConstantIndexOp>(expressionLocation, 0);
      Value one = builder.create<arith::ConstantIndexOp>(expressionLocation, 1);
      Value inputLength = builder.create<tensor::DimOp>(expressionLocation, lhs, zero);
      Value coefficientLength = builder.create<tensor::DimOp>(expressionLocation, rhs, zero);
      Value nonempty = builder.create<arith::CmpIOp>(expressionLocation, arith::CmpIPredicate::ugt,
                                                     coefficientLength, zero);
      Value covered = builder.create<arith::CmpIOp>(expressionLocation, arith::CmpIPredicate::uge,
                                                    inputLength, coefficientLength);
      Value valid = builder.create<arith::AndIOp>(expressionLocation, nonempty, covered);
      Value rawLength = builder.create<arith::AddIOp>(
          expressionLocation,
          builder.create<arith::SubIOp>(expressionLocation, inputLength, coefficientLength), one);
      dynamicSizes.push_back(
          builder.create<arith::SelectOp>(expressionLocation, valid, rawLength, zero));
    }
    Value init = builder.create<tensor::EmptyOp>(expressionLocation, outputType.getShape(),
                                                 elementType, dynamicSizes);

    Attribute numeric;
    ondsp::ProductAttr product;
    TypeAttr accumulator;
    ondsp::FixedAttr destination;
    ondsp::RoundingModeAttr rounding;
    ondsp::OverflowModeAttr overflow;
    if (kernel.ast.primaryResult().type == SourceType::F32) {
      numeric = ondsp::FpAttr::get(&context, elementType, *kernel.fpContract);
    } else {
      unsigned storageWidth = cast<IntegerType>(elementType).getWidth();
      unsigned fractionalBits = storageWidth - 1;
      unsigned accumulatorWidth = kernel.ast.result.accumulatorWidth;
      auto fixed =
          ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, fractionalBits);
      auto accumulatorType = ondsp::AccType::get(&context, builder.getIntegerType(accumulatorWidth),
                                                 fractionalBits * 2, ondsp::Signedness::Signed,
                                                 *kernel.updateOverflow);
      numeric = fixed;
      product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
      accumulator = TypeAttr::get(accumulatorType);
      destination = fixed;
      rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
      overflow = ondsp::OverflowModeAttr::get(&context, *kernel.destinationOverflow);
    }

    Value result;
    if (isConv1D) {
      ir::Conv1DMode mode = kernel.ast.result.kind == ReductionKind::Convolution
                                ? ir::Conv1DMode::Convolution
                                : ir::Conv1DMode::Correlation;
      result = builder.create<ir::Conv1DOp>(expressionLocation, outputType, lhs, rhs, init, mode,
                                            numeric, product, accumulator, destination, rounding,
                                            overflow);
    } else if (isFirDecimate) {
      result = builder.create<ir::FirDecimateOp>(
          expressionLocation, outputType, lhs, rhs, init,
          builder.getI64IntegerAttr(kernel.ast.result.factor), numeric, product, accumulator,
          destination, rounding, overflow);
    } else if (isFirInterpolate) {
      result = builder.create<ir::FirInterpolateOp>(
          expressionLocation, outputType, lhs, rhs, init,
          builder.getI64IntegerAttr(kernel.ast.result.factor), numeric, product, accumulator,
          destination, rounding, overflow);
    } else {
      ir::FirBoundaryMode boundary = kernel.ast.result.boundary == "full"
                                         ? ir::FirBoundaryMode::Full
                                         : ir::FirBoundaryMode::Valid;
      result = builder.create<ir::FirFilterOp>(expressionLocation, outputType, lhs, rhs, init,
                                               Value(), boundary, numeric, product, accumulator,
                                               destination, rounding, overflow);
    }
    builder.create<func::ReturnOp>(expressionLocation, result);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }

  if (kernel.ast.primaryResult().type != SourceType::F32) {
    unsigned storageWidth = cast<IntegerType>(elementType).getWidth();
    unsigned fractionalBits = storageWidth - 1;
    unsigned accumulatorWidth = kernel.ast.result.accumulatorWidth;
    bool rawHigh = kernel.ast.result.product == "raw_high";
    unsigned accumulatorFractionalBits =
        rawHigh ? 2 * fractionalBits - storageWidth : fractionalBits * 2;
    auto numeric =
        ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, fractionalBits);
    auto product = ondsp::ProductAttr::get(&context, rawHigh ? ondsp::ProductSelection::HighRaw
                                                             : ondsp::ProductSelection::Full);
    auto accumulatorType = ondsp::AccType::get(&context, builder.getIntegerType(accumulatorWidth),
                                               accumulatorFractionalBits, ondsp::Signedness::Signed,
                                               *kernel.updateOverflow);
    Value accumulator;
    if (kernel.ast.result.kind == ReductionKind::Dot) {
      IntegerAttr gain = kernel.ast.result.gainBound
                             ? builder.getI64IntegerAttr(kernel.ast.result.gainBound)
                             : IntegerAttr();
      accumulator = builder
                        .create<ir::DotOp>(expressionLocation, accumulatorType, lhs, rhs, numeric,
                                           product, gain)
                        .getResult();
    } else
      accumulator =
          builder.create<ir::FirOp>(expressionLocation, accumulatorType, lhs, rhs, numeric, product)
              .getResult();
    Value result;
    if (rawHigh) {
      // The raw high half reads one bit below Q31, and an export never raises
      // the fractional position: the identity export into the i64 carrier, one
      // exact doubling there, then the declared narrowing.
      auto i64 = builder.getI64Type();
      auto carrier = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, i64,
                                           accumulatorFractionalBits);
      Value wide =
          builder.create<ondsp::AccExportOp>(expressionLocation, i64, accumulator, carrier,
                                             *kernel.rounding, *kernel.destinationOverflow);
      Value one = builder.create<arith::ConstantIntOp>(expressionLocation, 1, i64);
      Value doubled = builder.create<arith::ShLIOp>(expressionLocation, wide, one);
      result = builder.create<ondsp::RoundShiftOp>(
          expressionLocation, elementType, doubled,
          ondsp::ScaleAttr::get(&context, 0, 0, *kernel.rounding, *kernel.destinationOverflow,
                                elementType));
    } else {
      result =
          builder.create<ondsp::AccExportOp>(expressionLocation, elementType, accumulator, numeric,
                                             *kernel.rounding, *kernel.destinationOverflow);
    }
    builder.create<func::ReturnOp>(expressionLocation, result);
  } else {
    auto numeric = ondsp::FpAttr::get(&context, elementType, *kernel.fpContract);
    Value result;
    if (kernel.ast.result.kind == ReductionKind::Dot)
      result = builder.create<ir::DotOp>(expressionLocation, elementType, lhs, rhs, numeric,
                                         ondsp::ProductAttr(), IntegerAttr());
    else
      result = builder.create<ir::FirOp>(expressionLocation, elementType, lhs, rhs, numeric,
                                         ondsp::ProductAttr());
    builder.create<func::ReturnOp>(expressionLocation, result);
  }

  module->push_back(function);
  if (failed(verify(*module)))
    return {};
  return module;
}

} // namespace

OwningOpRef<ModuleOp> compileOxSource(llvm::StringRef sourceName, llvm::StringRef source,
                                      MLIRContext &context, llvm::raw_ostream &diagnosticsOutput) {
  Diagnostics diagnostics(sourceName, source, diagnosticsOutput);
  Lexer lexer(source, diagnostics);
  Parser parser(lexer, diagnostics);
  std::optional<KernelAst> ast = parser.parse();
  if (!ast || diagnostics.failed())
    return {};

  // Each callee is checked once against its own signature, at its own source
  // location. Instantiation then only has to match arguments to that
  // signature, and the errors a user sees name the function that is wrong.
  for (const KernelAst &callee : parser.getCallees()) {
    if (!checkKernel(KernelAst(callee), diagnostics) || diagnostics.failed())
      return {};
  }

  std::optional<CheckedKernel> checked = checkKernel(std::move(*ast), diagnostics);
  if (!checked || diagnostics.failed())
    return {};
  return generateModule(*checked, sourceName, context);
}

} // namespace ondrix::frontend
