#include "ondrix/Frontend/OxFrontend.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
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
  LeftParen,
  RightParen,
  LeftBracket,
  RightBracket,
  Comma,
  Colon,
  Equal,
  Minus,
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
      do {
        advance();
      } while (offset < source.size() && std::isdigit(static_cast<unsigned char>(source[offset])));
      return {TokenKind::Integer, source.slice(begin, offset), start};
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

enum class SourceType { Q15, Q31, F32, ComplexQ15 };

enum class ContainerKind { Scalar, Buffer, Tensor, Constexpr };

enum class ReductionKind {
  Dot,
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
  Log2,
  Exp2
};

// The elementwise family: every member is one exact integer expression plus
// one declared boundary, and each has a statically known result shape, which
// is what lets them appear as operands of one another.
static bool isElementwiseKind(ReductionKind kind) {
  return kind == ReductionKind::Add || kind == ReductionKind::Sub || kind == ReductionKind::Mult ||
         kind == ReductionKind::Abs || kind == ReductionKind::Negate ||
         kind == ReductionKind::Offset || kind == ReductionKind::Shift;
}

static bool isBinaryElementwiseKind(ReductionKind kind) {
  return kind == ReductionKind::Add || kind == ReductionKind::Sub || kind == ReductionKind::Mult;
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
  return isFftKind(kind) || kind == ReductionKind::Magnitude || kind == ReductionKind::Phase;
}

// The set whose members may hold one another as operands. Its checker
// carries an element type and extent from operand to result, so a member has
// to have a statically derivable result shape.
static bool isComposableKind(ReductionKind kind) {
  return isFftComposableKind(kind) || isElementwiseKind(kind);
}

static bool isUnaryTensorKind(ReductionKind kind) {
  return kind == ReductionKind::Dct || kind == ReductionKind::MovingAverage ||
         kind == ReductionKind::Gain || kind == ReductionKind::Rms || kind == ReductionKind::Sine ||
         kind == ReductionKind::Cosine || kind == ReductionKind::CicDecimate ||
         kind == ReductionKind::Log2 || kind == ReductionKind::Exp2 ||
         kind == ReductionKind::Goertzel;
}

static bool isUnaryKind(ReductionKind kind) {
  return isFftComposableKind(kind) || isUnaryTensorKind(kind) ||
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

struct BuiltinCallAst {
  ReductionKind kind;
  std::vector<ExpressionAst> operands;
  uint64_t accumulatorWidth = 0;
  bool accumulatorAuto = false;
  std::string updateOverflow;
  std::string rounding;
  std::string destinationOverflow;
  std::string stateRounding;
  std::string stateOverflow;
  std::string fpContract;
  std::string boundary;
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
  SourcePosition position;
};

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
    do {
      std::optional<ParameterAst> parameter = parseParameter();
      if (!parameter)
        return std::nullopt;
      kernel.parameters.push_back(std::move(*parameter));
      if (current.kind != TokenKind::Comma)
        break;
      advance();
    } while (true);

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
    while (current.kind == TokenKind::Identifier && current.spelling != "return" &&
           next.kind == TokenKind::Equal) {
      Token name = current;
      bool collides = bindingsByName.contains(name.spelling) ||
                      llvm::any_of(kernel.parameters, [&](const ParameterAst &parameter) {
                        return parameter.name == name.spelling;
                      });
      if (collides) {
        diagnostics.error(name.position, llvm::Twine("local '") + name.spelling +
                                             "' collides with an existing name");
        return std::nullopt;
      }
      advance();
      advance();
      std::optional<BuiltinCallAst> bound = parseBuiltinCall(policyType);
      if (!bound)
        return std::nullopt;
      bindingsByName[name.spelling] = bindings.size();
      bindings.push_back(
          Binding{name.spelling.str(), ExpressionAst(std::move(*bound)), name.position, false});
    }
    if (!expectIdentifier("return", "expected a single return statement"))
      return std::nullopt;
    std::optional<BuiltinCallAst> result = parseBuiltinCall(policyType);
    if (!result)
      return std::nullopt;
    kernel.result = std::move(*result);
    caller = nullptr;
    if (current.kind != TokenKind::Eof && !isIdentifier("def")) {
      diagnostics.error(current.position,
                        "a function body is a single return statement; expected 'def' or "
                        "end of file");
      return std::nullopt;
    }
    for (const Binding &binding : bindings) {
      if (!binding.consumed) {
        diagnostics.error(binding.position, llvm::Twine("local '") + binding.name +
                                                "' is never consumed by a later statement");
        return std::nullopt;
      }
    }
    return kernel;
  }

  std::optional<BuiltinCallAst> parseBuiltinCall(SourceType policyType) {
    if (current.kind == TokenKind::Identifier && next.kind == TokenKind::LeftParen &&
        calleesByName.contains(current.spelling))
      return parseCalleeInstantiation(policyType);
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
        !isIdentifier("phase") && !isIdentifier("dct") && !isIdentifier("moving_average") &&
        !isIdentifier("gain") && !isIdentifier("rms") && !isIdentifier("sine") &&
        !isIdentifier("cosine") && !isIdentifier("matmul") && !isIdentifier("lms") &&
        !isIdentifier("lowpass") && !isIdentifier("cic_decimate") && !isIdentifier("add") &&
        !isIdentifier("sub") && !isIdentifier("mult") && !isIdentifier("abs") &&
        !isIdentifier("negate") && !isIdentifier("offset") && !isIdentifier("shift") &&
        !isIdentifier("log2") && !isIdentifier("exp2")) {
      diagnostics.error(current.position,
                        "expected dot(...), fir(...), fir_filter(...), fir_decimate(...), "
                        "fir_interpolate(...), fir_stream(...), sos_df2_fixed(...), "
                        "sos_tdf2(...), goertzel(...), "
                        "convolution(...), correlation(...), butterfly(...), cfft(...), or "
                        "icfft(...), rfft(...), irfft(...), magnitude(...), phase(...), dct(...), "
                        "moving_average(...), gain(...), rms(...), sine(...), cosine(...), "
                        "matmul(...), lms(...), cic_decimate(...), a lowpass/hamming/hann/"
                        "blackman/kaiser design, or an "
                        "elementwise add/sub/mult/abs/negate/offset/shift builtin expression");
      return std::nullopt;
    }
    if (isIdentifier("dot"))
      call.kind = ReductionKind::Dot;
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
    else if (isIdentifier("cic_decimate"))
      call.kind = ReductionKind::CicDecimate;
    else if (isIdentifier("add"))
      call.kind = ReductionKind::Add;
    else if (isIdentifier("sub"))
      call.kind = ReductionKind::Sub;
    else if (isIdentifier("mult"))
      call.kind = ReductionKind::Mult;
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
      std::optional<ExpressionAst> operand = parseComposedOperand();
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      if (!expect(TokenKind::RightParen, "expected ')' after FFT operand"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::Phase) {
      // The phase contract admits exactly one tie rule, so unlike magnitude
      // there is no rounding choice to expose at the call site.
      std::optional<ExpressionAst> operand = parseComposedOperand();
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      if (!expect(TokenKind::RightParen, "expected ')' after phase operand"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::Magnitude) {
      std::optional<ExpressionAst> operand = parseComposedOperand();
      if (!operand)
        return std::nullopt;
      call.operands.push_back(std::move(*operand));
      if (current.kind == TokenKind::Comma) {
        // The magnitude contract admits a declared root rounding mode;
        // omission keeps the nearest_even default. The parameter names the
        // ROOT boundary — the sum of squares stays exact.
        if (!expect(TokenKind::Comma, "expected ',' before root rounding policy") ||
            !expectIdentifier("root_rounding", "expected root rounding policy") ||
            !expect(TokenKind::Equal, "expected '=' after root_rounding"))
          return std::nullopt;
        auto rounding = parseIdentifier("expected rounding mode");
        if (!rounding)
          return std::nullopt;
        call.rounding = rounding->spelling.str();
      }
      if (!expect(TokenKind::RightParen, "expected ')' after magnitude operand"))
        return std::nullopt;
      return call;
    }
    if (isElementwiseKind(call.kind)) {
      std::optional<ExpressionAst> lhs = parseComposedOperand();
      if (!lhs)
        return std::nullopt;
      call.operands.push_back(std::move(*lhs));
      if (isBinaryElementwiseKind(call.kind)) {
        if (!expect(TokenKind::Comma, "expected ',' before the second elementwise operand"))
          return std::nullopt;
        std::optional<ExpressionAst> rhs = parseComposedOperand();
        if (!rhs)
          return std::nullopt;
        call.operands.push_back(std::move(*rhs));
      }
      if (call.kind == ReductionKind::Offset && !parseNamedInteger("bias", call.bias))
        return std::nullopt;
      if (call.kind == ReductionKind::Shift && !parseNamedInteger("amount", call.amount))
        return std::nullopt;
      // Both boundary choices are optional and both default to the rule the
      // rest of the language uses: unbiased, non-wrapping.
      if (current.kind == TokenKind::Comma && next.spelling == "rounding") {
        if (!expect(TokenKind::Comma, "expected ',' before rounding policy") ||
            !expectIdentifier("rounding", "expected rounding policy") ||
            !expect(TokenKind::Equal, "expected '=' after rounding"))
          return std::nullopt;
        auto rounding = parseIdentifier("expected rounding mode");
        if (!rounding)
          return std::nullopt;
        call.rounding = rounding->spelling.str();
      }
      if (current.kind == TokenKind::Comma) {
        if (!expect(TokenKind::Comma, "expected ',' before overflow policy") ||
            !expectIdentifier("overflow", "expected overflow policy") ||
            !expect(TokenKind::Equal, "expected '=' after overflow"))
          return std::nullopt;
        auto overflow = parseIdentifier("expected overflow mode");
        if (!overflow)
          return std::nullopt;
        call.destinationOverflow = overflow->spelling.str();
      }
      if (!expect(TokenKind::RightParen, "expected ')' after elementwise expression"))
        return std::nullopt;
      return call;
    }
    auto lhs = parseIdentifier("expected builtin operand");
    if (!lhs)
      return std::nullopt;
    call.operands.push_back(resolveOperand(*lhs));
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
      if (!third || !expect(TokenKind::RightParen,
                            llvm::Twine("expected ')' after ") + builtin + " operands"))
        return std::nullopt;
      call.operands.push_back(resolveOperand(*rhs));
      call.operands.push_back(resolveOperand(*third));
      if (call.kind == ReductionKind::FirStream) {
        applyDefaultFixedPolicy(call);
      }
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
        // The rms contract admits a declared root rounding mode while the
        // mean boundary stays nearest even; omission keeps the nearest_even
        // default. The parameter names the specific boundary (`root_`)
        // because rms carries two rounding boundaries. Bindings must expose
        // every choice their contract admits.
        if (!expect(TokenKind::Comma, "expected ',' before root rounding policy") ||
            !expectIdentifier("root_rounding", "expected root rounding policy") ||
            !expect(TokenKind::Equal, "expected '=' after root_rounding"))
          return std::nullopt;
        auto rounding = parseIdentifier("expected rounding mode");
        if (!rounding)
          return std::nullopt;
        call.rounding = rounding->spelling.str();
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
      }
      if (!expect(TokenKind::RightParen, "expected ')' after matmul expression"))
        return std::nullopt;
      return call;
    }
    if (call.kind == ReductionKind::FirFilter) {
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
      } else {
        applyDefaultFixedPolicy(call);
      }
      if (!expect(TokenKind::RightParen,
                  llvm::Twine("expected ')' after ") + operation + " expression"))
        return std::nullopt;
      return call;
    } else if (current.kind == TokenKind::RightParen && policyType != SourceType::F32) {
      applyDefaultFixedPolicy(call);
    } else if (!expect(TokenKind::Comma, "expected ',' before numeric policy")) {
      return std::nullopt;
    }

    if (policyType != SourceType::F32) {
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
  // An operand of a composable builtin is a name or another composable
  // expression; the whole family shares one operand grammar so that adding a
  // member does not add a nesting rule.
  std::optional<ExpressionAst> parseComposedOperand() {
    if (current.kind == TokenKind::Identifier && next.kind == TokenKind::LeftParen &&
        calleesByName.contains(current.spelling)) {
      // Inside a chain the composition checker establishes element types, so
      // the call is not additionally tied to the function's result type.
      std::optional<BuiltinCallAst> instantiated = parseCalleeInstantiation(std::nullopt);
      if (!instantiated)
        return std::nullopt;
      return ExpressionAst(std::move(*instantiated));
    }
    bool isNestedBuiltin = isIdentifier("cfft") || isIdentifier("icfft") || isIdentifier("rfft") ||
                           isIdentifier("irfft") || isIdentifier("magnitude") ||
                           isIdentifier("add") || isIdentifier("sub") || isIdentifier("mult") ||
                           isIdentifier("abs") || isIdentifier("negate") ||
                           isIdentifier("offset") || isIdentifier("shift");
    if (!isNestedBuiltin || next.kind != TokenKind::LeftParen) {
      auto parameter = parseIdentifier("expected an operand name or nested expression");
      if (!parameter)
        return std::nullopt;
      return resolveOperand(*parameter);
    }
    std::optional<BuiltinCallAst> nested = parseBuiltinCall(SourceType::Q15);
    if (!nested)
      return std::nullopt;
    return ExpressionAst(std::move(*nested));
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
    if (value->spelling.getAsInteger(10, magnitude) ||
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
        if (parsedExtent->spelling.getAsInteger(10, value) ||
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
    return instantiateCall(callee.result, substitution, name.position);
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
    result.accumulatorWidth = 40;
    result.updateOverflow = "wrap";
    result.stateRounding = "nearest_ties_positive";
    result.stateOverflow = "saturate";
    result.rounding = "nearest_ties_positive";
    result.destinationOverflow = "saturate";
  }

  bool parseFixedPolicy(BuiltinCallAst &result) {
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
    if (width->spelling.getAsInteger(10, result.accumulatorWidth)) {
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
  };

  // Every reference to a local instantiates a fresh deep copy of the bound
  // call tree. The Ondrix tensor operations the copies produce are Pure, so
  // the pipeline's canonicalize/cse collapses them back to one evaluation.
  ExpressionAst resolveOperand(const Token &token) {
    auto binding = bindingsByName.find(token.spelling);
    if (binding == bindingsByName.end())
      return ExpressionAst(token.spelling.str(), token.position);
    Binding &bound = bindings[binding->second];
    bound.consumed = true;
    return ExpressionAst(bound.expression);
  }

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
  if (ast.result.operands.size() != expectedOperandCount) {
    diagnostics.error(ast.result.position, "builtin operand count does not match its contract");
    return std::nullopt;
  }
  if (!isComposableKind(ast.result.kind) &&
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
  if (!isComposableKind(ast.result.kind) && ast.parameters.size() != expectedParameterCount) {
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
  for (const ParameterAst &parameter : ast.parameters) {
    if (!parameterNames.insert(parameter.name).second) {
      diagnostics.error(parameter.position,
                        llvm::Twine("duplicate parameter '") + parameter.name + "'");
      return std::nullopt;
    }
    if (!isComposableKind(ast.result.kind) && parameter.type != ast.primaryResult().type) {
      diagnostics.error(parameter.position,
                        "parameter element types must match the kernel result type");
      return std::nullopt;
    }
    if (!verifyPositiveShape(parameter.shape, "parameter", parameter.position))
      return std::nullopt;
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
  if (!isComposableKind(ast.result.kind) && (!lhsName || !parameterNames.contains(*lhsName))) {
    diagnostics.error(ast.result.position, llvm::Twine("unknown builtin operand '") +
                                               (lhsName ? *lhsName : "<nested call>") + "'");
    return std::nullopt;
  }
  if (!isUnaryKind(ast.result.kind) && !isComposableKind(ast.result.kind) &&
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
  if (ast.result.kind == ReductionKind::FirStream) {
    if (ast.results.size() != 2 || !ast.primaryResult().tensor || !ast.results[1].tensor ||
        ast.primaryResult().type != SourceType::Q15 || !lhsParameter || !rhsParameter ||
        !thirdParameter || llvm::any_of(ast.parameters, [](const ParameterAst &parameter) {
          return !parameter.isTensor();
        })) {
      diagnostics.error(
          ast.result.position,
          "fir_stream requires three Q15 tensor parameters and two Q15 tensor results");
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
    else {
      ast.result.accumulatorWidth =
          inferQ15FullAccumulatorWidth(static_cast<uint64_t>(coefficientExtent));
      return CheckedKernel{std::move(ast), ondsp::OverflowMode::Wrap,
                           ondsp::RoundingMode::NearestEven, ondsp::OverflowMode::Saturate,
                           std::nullopt};
    }
    return std::nullopt;
  }
  if (ast.result.kind == ReductionKind::SosDf2Fixed) {
    if (ast.results.size() != 2 || !ast.primaryResult().tensor || !ast.results[1].tensor ||
        ast.primaryResult().type != SourceType::Q15 || !lhsParameter || !rhsParameter ||
        !thirdParameter || !fourthParameter || constexprCount != 0 ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isTensor(); })) {
      diagnostics.error(
          ast.result.position,
          "sos_df2_fixed requires four Q15 tensor parameters and two Q15 tensor results");
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
    if (ast.result.accumulatorWidth != 40) {
      diagnostics.error(ast.result.position,
                        "the executable Q15 SOS profile requires exact accumulator width 40");
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
    if (ast.results.size() != 1 || !ast.primaryResult().tensor ||
        (ast.primaryResult().type != SourceType::Q15 && !isFloat) || !lhsParameter ||
        !rhsParameter || constexprCount != 0 ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isTensor(); })) {
      diagnostics.error(ast.result.position,
                        "matmul requires two Q15 or f32 tensor parameters and a matching "
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
    return CheckedKernel{std::move(ast), std::nullopt, ondsp::RoundingMode::NearestEven,
                         std::nullopt, std::nullopt};
  }
  if (ast.result.kind == ReductionKind::Lms) {
    bool isFloat = ast.primaryResult().type == SourceType::F32;
    if (ast.results.size() != 2 || !ast.primaryResult().tensor || !ast.results[1].tensor ||
        (ast.primaryResult().type != SourceType::Q15 && !isFloat) || !lhsParameter ||
        !rhsParameter || !thirdParameter || constexprCount != 0 ||
        llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return !parameter.isTensor(); })) {
      diagnostics.error(
          ast.result.position,
          "lms requires three Q15 or f32 tensor parameters and two matching tensor results");
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
    if (ast.result.stepSize < 0 || ast.result.stepSize > 32767) {
      diagnostics.error(ast.result.position,
                        "lms step size must be a raw signed Q1.15 value in [0, 32767]");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), std::nullopt, ondsp::RoundingMode::NearestEven,
                         std::nullopt, std::nullopt};
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
  if (isComposableKind(ast.result.kind)) {
    struct ComposedType {
      SourceType elementType;
      int64_t extent;
    };
    llvm::StringMap<unsigned> parameterUses;
    std::function<std::optional<ComposedType>(ExpressionAst &)> checkComposedExpression;
    std::function<std::optional<ComposedType>(BuiltinCallAst &)> checkComposedCall;
    checkComposedExpression = [&](ExpressionAst &expression) -> std::optional<ComposedType> {
      if (expression.isParameterReference()) {
        auto parameter = parametersByName.find(expression.parameter);
        if (parameter == parametersByName.end()) {
          diagnostics.error(expression.position,
                            llvm::Twine("unknown operand '") + expression.parameter + "'");
          return std::nullopt;
        }
        ++parameterUses[expression.parameter];
        const ParameterAst *value = parameter->second;
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

      return checkComposedCall(*expression.call);
    };
    // A fir_filter stage may feed the FFT chain: static Q15 tensors, the
    // valid boundary, and the executable Q15 export profile, explicitly
    // declared. Its coefficients are a tensor parameter or a design builtin.
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
      if (!input->isTensor() || input->type != SourceType::Q15 || !hasRank(input->shape, 1) ||
          !getRankOneExtent(input->shape)) {
        diagnostics.error(inputOperand.position,
                          "a composed fir_filter currently requires a static rank-1 Q15 tensor "
                          "input");
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
        if (!coefficients->isTensor() || coefficients->type != SourceType::Q15 ||
            !hasRank(coefficients->shape, 1) || !getRankOneExtent(coefficients->shape)) {
          diagnostics.error(coefficientOperand.position,
                            "composed fir_filter coefficients currently require a static rank-1 "
                            "Q15 tensor parameter or a coefficient design");
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
                          "Q15 tensor parameter or a coefficient design");
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
      // the tap count is already static here, so the inference is identical.
      if (call.accumulatorAuto) {
        call.accumulatorAuto = false;
        call.accumulatorWidth = inferQ15FullAccumulatorWidth(static_cast<uint64_t>(tapCount));
      } else if (call.accumulatorWidth != 40) {
        diagnostics.error(call.position,
                          "the executable Q15 profile requires exact accumulator width 40");
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
      return ComposedType{SourceType::Q15, inputExtent - tapCount + 1};
    };
    // The elementwise family: Q15 in, Q15 out, extents equal. Only the two
    // boundary attributes vary between members, so one checker covers all
    // seven and a new member cannot forget a rule.
    auto checkElementwise = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      std::optional<ComposedType> lhs = checkComposedExpression(call.operands.front());
      if (!lhs)
        return std::nullopt;
      if (lhs->elementType != SourceType::Q15) {
        diagnostics.error(call.position, "elementwise builtins require Q15 operand elements");
        return std::nullopt;
      }
      if (lhs->extent < 1 || lhs->extent > 4096) {
        diagnostics.error(call.position,
                          "elementwise builtins currently require an extent in [1, 4096]");
        return std::nullopt;
      }
      if (isBinaryElementwiseKind(call.kind)) {
        std::optional<ComposedType> rhs = checkComposedExpression(call.operands[1]);
        if (!rhs)
          return std::nullopt;
        if (rhs->elementType != SourceType::Q15 || rhs->extent != lhs->extent) {
          diagnostics.error(call.position,
                            "binary elementwise builtins require operands of the same Q15 extent");
          return std::nullopt;
        }
      }
      if (call.kind == ReductionKind::Offset && (call.bias < -32768 || call.bias > 32767)) {
        diagnostics.error(call.position,
                          "offset bias must be a raw signed Q1.15 value in [-32768, 32767]");
        return std::nullopt;
      }
      if (call.kind == ReductionKind::Shift && (call.amount < -15 || call.amount > 15)) {
        diagnostics.error(call.position, "shift amount must lie in [-15, 15]");
        return std::nullopt;
      }
      if (call.rounding.empty())
        call.rounding = "nearest_ties_positive";
      if (call.destinationOverflow.empty())
        call.destinationOverflow = "saturate";
      if (!parseRounding(call.rounding)) {
        diagnostics.error(call.position,
                          llvm::Twine("unsupported rounding mode '") + call.rounding + "'");
        return std::nullopt;
      }
      if (!parseOverflow(call.destinationOverflow)) {
        diagnostics.error(call.position, llvm::Twine("unsupported overflow mode '") +
                                             call.destinationOverflow + "'");
        return std::nullopt;
      }
      return *lhs;
    };
    checkComposedCall = [&](BuiltinCallAst &call) -> std::optional<ComposedType> {
      if (call.kind == ReductionKind::FirFilter)
        return checkNestedFirFilter(call);
      if (isElementwiseKind(call.kind))
        return checkElementwise(call);
      if (!isFftComposableKind(call.kind) || call.operands.size() != 1) {
        diagnostics.error(call.position,
                          "nested calls are currently supported only by unary FFT-family builtins "
                          "and a fir_filter input stage");
        return std::nullopt;
      }
      std::optional<ComposedType> input = checkComposedExpression(call.operands.front());
      if (!input)
        return std::nullopt;

      if (call.kind == ReductionKind::Phase) {
        if (input->elementType != SourceType::ComplexQ15) {
          diagnostics.error(call.position, "phase requires complex_q15 operand elements");
          return std::nullopt;
        }
        if (input->extent < 1 || input->extent > 4096) {
          diagnostics.error(call.position,
                            "phase currently requires an operand extent in [1, 4096]");
          return std::nullopt;
        }
        return ComposedType{SourceType::Q15, input->extent};
      }
      if (call.kind == ReductionKind::Magnitude) {
        if (input->elementType != SourceType::ComplexQ15) {
          diagnostics.error(call.position, "magnitude requires complex_q15 operand elements");
          return std::nullopt;
        }
        if (input->extent < 1 || input->extent > 4096) {
          diagnostics.error(call.position,
                            "magnitude currently requires an operand extent in [1, 4096]");
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
        return ComposedType{SourceType::Q15, input->extent};
      }
      if (isCfftKind(call.kind)) {
        if (input->elementType != SourceType::ComplexQ15) {
          diagnostics.error(call.position, "cfft and icfft require complex_q15 operand elements");
          return std::nullopt;
        }
        if (input->extent != 4 && input->extent != 8) {
          diagnostics.error(call.position, "cfft currently supports only four or eight points");
          return std::nullopt;
        }
        return *input;
      }
      if (call.kind == ReductionKind::Rfft) {
        if (input->elementType != SourceType::Q15) {
          diagnostics.error(call.position, "rfft requires Q15 real operand elements");
          return std::nullopt;
        }
        if (input->extent < 8 || input->extent > 64 || !llvm::isPowerOf2_64(input->extent)) {
          diagnostics.error(call.position,
                            "rfft currently supports power-of-two extents in [8, 64]");
          return std::nullopt;
        }
        return ComposedType{SourceType::ComplexQ15, input->extent / 2 + 1};
      }
      if (input->elementType != SourceType::ComplexQ15) {
        diagnostics.error(call.position, "irfft requires complex_q15 Hermitian operand elements");
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
         ast.primaryResult().type != SourceType::F32) ||
        !lhsParameter || !rhsParameter || !lhsParameter->isTensor() || !rhsParameter->isTensor()) {
      diagnostics.error(ast.result.position,
                        "fir_decimate requires Q15 or f32 tensor input, coefficients, and result");
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
    // The Q15 goertzel energy is tensor<1xi64>, a storage width no source
    // type names, so only the f32 profile has a spelling here.
    if (ast.result.kind == ReductionKind::Goertzel && !isFloat) {
      diagnostics.error(ast.result.position, "goertzel currently binds only the f32 profile");
      return std::nullopt;
    }
    if (constexprCount != 0 || !ast.primaryResult().tensor || !lhsParameter ||
        !lhsParameter->isTensor() ||
        (ast.primaryResult().type != SourceType::Q15 && !(admitsFloat && isFloat))) {
      diagnostics.error(ast.result.position,
                        llvm::Twine(builtin) + (admitsFloat ? " requires a Q15 or f32 tensor input "
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
      // The source type system only names the i16 storage: the declared
      // result reads as q15 while the emitted operation carries the
      // derived frac = 14 - log2(N) output reading in its attribute. The
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
        auto [constant, refusal] = roundRationalToF32(ast.result.gain, ast.result.fpConstantDen);
        if (refusal != RationalRefusal::None) {
          diagnostics.error(ast.result.position,
                            llvm::Twine("f32 gain constant: ") + describeRationalRefusal(refusal));
          return std::nullopt;
        }
        ast.result.fpConstant = constant;
      } else if (ast.result.gain < std::numeric_limits<int16_t>::min() ||
                 ast.result.gain > std::numeric_limits<int16_t>::max()) {
        diagnostics.error(ast.result.position,
                          "gain constant must be a raw signed Q1.15 value in [-32768, 32767]");
        return std::nullopt;
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
    ondsp::RoundingMode rounding = isGain || isCic ? ondsp::RoundingMode::NearestTiesPositive
                                                   : ondsp::RoundingMode::NearestEven;
    if (!ast.result.rounding.empty()) {
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
    return CheckedKernel{std::move(ast), std::nullopt, rounding, std::nullopt, std::nullopt};
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

  uint64_t requiredAccumulatorWidth = ast.primaryResult().type == SourceType::Q15 ? 40 : 64;
  if (!ast.result.accumulatorAuto && ast.result.accumulatorWidth != requiredAccumulatorWidth) {
    diagnostics.error(ast.result.position,
                      ast.primaryResult().type == SourceType::Q15
                          ? "the executable Q15 profile requires exact accumulator width 40"
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
  context.loadDialect<arith::ArithDialect, bufferization::BufferizationDialect, func::FuncDialect,
                      memref::MemRefDialect, tensor::TensorDialect, ir::OndrixDialect,
                      ondsp::OndspDialect>();
  OpBuilder builder(&context);
  Location kernelLocation = getLocation(context, sourceName, kernel.ast.position);
  OwningOpRef<ModuleOp> module = ModuleOp::create(kernelLocation);

  auto getStorageType = [&](SourceType type) -> Type {
    if (type == SourceType::Q15)
      return builder.getI16Type();
    if (type == SourceType::Q31 || type == SourceType::ComplexQ15)
      return builder.getI32Type();
    return builder.getF32Type();
  };
  Type elementType = getStorageType(kernel.ast.primaryResult().type);
  auto materializeShape = [](llvm::ArrayRef<std::optional<int64_t>> shape) {
    SmallVector<int64_t> dimensions;
    dimensions.reserve(shape.size());
    for (std::optional<int64_t> extent : shape)
      dimensions.push_back(extent.value_or(ShapedType::kDynamic));
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
      inputTypes.push_back(
          RankedTensorType::get(materializeShape(parameter.shape), parameterElementType));
    else
      inputTypes.push_back(
          MemRefType::get(materializeShape(parameter.shape), parameterElementType));
  }
  SmallVector<Type> resultTypes;
  resultTypes.reserve(kernel.ast.results.size());
  for (const ResultTypeAst &result : kernel.ast.results) {
    Type resultElementType = getStorageType(result.type);
    resultTypes.push_back(result.tensor ? Type(RankedTensorType::get(materializeShape(result.shape),
                                                                     resultElementType))
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
      arguments.insert({parameter.name, entry->getArgument(argumentIndex++)});
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
  if (isComposableKind(kernel.ast.result.kind)) {
    auto layout = ondsp::CxLayoutAttr::get(&context, ondsp::ComplexLayout::PackedI16ImagHiRealLo);
    auto i16 = builder.getI16Type();
    auto numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, i16, 15);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto productScale = ondsp::ScaleAttr::get(&context, 0, 15, ondsp::RoundingMode::NearestEven,
                                              ondsp::OverflowMode::Saturate, i16);
    auto outputScale = ondsp::ScaleAttr::get(&context, 0, 1, ondsp::RoundingMode::NearestEven,
                                             ondsp::OverflowMode::Saturate, i16);
    std::function<Value(const BuiltinCallAst &)> emitComposedCall =
        [&](const BuiltinCallAst &call) -> Value {
      if (call.kind == ReductionKind::FirFilter) {
        // The composed filter stage: sema pinned Q15 elements, the valid
        // boundary, static extents, and the explicit width-40 profile.
        Location callLocation = getLocation(context, sourceName, call.position);
        Value signal = arguments.lookup(call.operands[0].parameter);
        const ExpressionAst &coefficientOperand = call.operands[1];
        Value coefficients;
        if (coefficientOperand.isParameterReference()) {
          coefficients = arguments.lookup(coefficientOperand.parameter);
        } else {
          const BuiltinCallAst &design = *coefficientOperand.call;
          Location designLocation = getLocation(context, sourceName, design.position);
          auto designType = RankedTensorType::get({design.taps}, i16);
          switch (design.kind) {
          case ReductionKind::Hamming:
            coefficients = builder.create<ir::WindowHammingOp>(designLocation, designType, numeric);
            break;
          case ReductionKind::Hann:
            coefficients = builder.create<ir::WindowHannOp>(designLocation, designType, numeric);
            break;
          case ReductionKind::Blackman:
            coefficients =
                builder.create<ir::WindowBlackmanOp>(designLocation, designType, numeric);
            break;
          case ReductionKind::Kaiser:
            coefficients = builder.create<ir::WindowKaiserOp>(
                designLocation, designType, builder.getI64IntegerAttr(design.betaNum),
                builder.getI64IntegerAttr(design.betaDen), numeric);
            break;
          default:
            coefficients = builder.create<ir::FirDesignWindowedSincOp>(
                designLocation, designType,
                ir::FirDesignResponseAttr::get(&context, ir::FirDesignResponse::Lowpass),
                builder.getI64IntegerAttr(design.cutoffNum),
                builder.getI64IntegerAttr(design.cutoffDen), numeric);
            break;
          }
        }
        int64_t inputExtent = cast<RankedTensorType>(signal.getType()).getDimSize(0);
        int64_t tapCount = cast<RankedTensorType>(coefficients.getType()).getDimSize(0);
        auto outputType = RankedTensorType::get({inputExtent - tapCount + 1}, i16);
        Value init = builder.create<tensor::EmptyOp>(callLocation, outputType.getShape(), i16);
        auto accumulatorType =
            ondsp::AccType::get(&context, builder.getIntegerType(call.accumulatorWidth), 30,
                                ondsp::Signedness::Signed, *parseOverflow(call.updateOverflow));
        return builder.create<ir::FirFilterOp>(
            callLocation, outputType, signal, coefficients, init, Value(),
            ir::FirBoundaryMode::Valid, numeric, product, TypeAttr::get(accumulatorType), numeric,
            ondsp::RoundingModeAttr::get(&context, *parseRounding(call.rounding)),
            ondsp::OverflowModeAttr::get(&context, *parseOverflow(call.destinationOverflow)));
      }
      const ExpressionAst &operand = call.operands.front();
      Value input = operand.isParameterReference() ? arguments.lookup(operand.parameter)
                                                   : emitComposedCall(*operand.call);
      auto inputType = cast<RankedTensorType>(input.getType());
      if (isElementwiseKind(call.kind)) {
        Location callLocation = getLocation(context, sourceName, call.position);
        auto rounding = ondsp::RoundingModeAttr::get(&context, *parseRounding(call.rounding));
        auto overflow =
            ondsp::OverflowModeAttr::get(&context, *parseOverflow(call.destinationOverflow));
        if (isBinaryElementwiseKind(call.kind)) {
          const ExpressionAst &second = call.operands[1];
          Value rhs = second.isParameterReference() ? arguments.lookup(second.parameter)
                                                    : emitComposedCall(*second.call);
          if (call.kind == ReductionKind::Add)
            return builder.create<ir::AddOp>(callLocation, inputType, input, rhs, numeric,
                                             overflow);
          if (call.kind == ReductionKind::Sub)
            return builder.create<ir::SubOp>(callLocation, inputType, input, rhs, numeric,
                                             overflow);
          return builder.create<ir::MultOp>(callLocation, inputType, input, rhs, numeric, rounding,
                                            overflow);
        }
        if (call.kind == ReductionKind::Abs)
          return builder.create<ir::AbsOp>(callLocation, inputType, input, numeric, overflow);
        if (call.kind == ReductionKind::Negate)
          return builder.create<ir::NegateOp>(callLocation, inputType, input, numeric, overflow);
        if (call.kind == ReductionKind::Offset)
          return builder.create<ir::OffsetOp>(callLocation, inputType, input,
                                              builder.getI64IntegerAttr(call.bias), numeric,
                                              overflow);
        return builder.create<ir::ShiftOp>(callLocation, inputType, input,
                                           builder.getI64IntegerAttr(call.amount), numeric,
                                           rounding, overflow);
      }
      Location callLocation = getLocation(context, sourceName, call.position);
      if (call.kind == ReductionKind::Phase) {
        // The result reading is the unsigned Q0.16 turn; the source type
        // system names only the i16 storage, so the binding supplies it —
        // the same projection log2/exp2 use.
        auto outputType = RankedTensorType::get({inputType.getDimSize(0)}, builder.getI16Type());
        auto turn =
            ondsp::FixedAttr::get(&context, ondsp::Signedness::Unsigned, builder.getI16Type(), 16);
        return builder.create<ir::CxPhaseOp>(
            callLocation, outputType, input, layout, numeric, turn,
            ondsp::RoundingModeAttr::get(&context, ondsp::RoundingMode::NearestEven));
      }
      if (call.kind == ReductionKind::Magnitude) {
        auto outputType = RankedTensorType::get({inputType.getDimSize(0)}, builder.getI16Type());
        // The declared root rounding was validated in sema; omission keeps
        // the nearest_even default.
        ondsp::RoundingMode rootRounding = ondsp::RoundingMode::NearestEven;
        if (!call.rounding.empty())
          rootRounding = *parseRounding(call.rounding);
        return builder.create<ir::CxMagnitudeOp>(
            callLocation, outputType, input, layout, numeric,
            ondsp::RoundingModeAttr::get(&context, rootRounding));
      }
      if (isCfftKind(call.kind)) {
        auto direction = ir::CfftDirectionAttr::get(&context, call.kind == ReductionKind::Cfft
                                                                  ? ir::CfftDirection::Forward
                                                                  : ir::CfftDirection::Inverse);
        return builder.create<ir::CfftOp>(callLocation, inputType, input, direction, layout,
                                          numeric, product, productScale, outputScale);
      }
      if (call.kind == ReductionKind::Rfft) {
        int64_t realExtent = inputType.getDimSize(0);
        auto outputType = RankedTensorType::get({realExtent / 2 + 1}, builder.getI32Type());
        return builder.create<ir::RfftOp>(callLocation, outputType, input, layout, numeric, product,
                                          productScale, outputScale);
      }
      int64_t realExtent = (inputType.getDimSize(0) - 1) * 2;
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
  if (isUnaryTensorKind(kernel.ast.result.kind)) {
    auto outputType = cast<RankedTensorType>(resultType);
    ondsp::FixedAttr numeric;
    ondsp::RoundingModeAttr rounding;
    if (!kernel.fpContract) {
      numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, 15);
      rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
    }
    Value result;
    if (kernel.ast.result.kind == ReductionKind::Dct) {
      if (kernel.fpContract) {
        auto fp = ondsp::FpAttr::get(&context, elementType, *kernel.fpContract);
        result = builder.create<ir::DctOp>(expressionLocation, outputType, lhs, fp, fp);
      } else {
        unsigned stageCount = llvm::Log2_64(outputType.getDimSize(0));
        auto outputNumeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType,
                                                   14 - stageCount);
        result =
            builder.create<ir::DctOp>(expressionLocation, outputType, lhs, numeric, outputNumeric);
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
      // The source type system names only the i16 storage, so the two
      // readings the contract distinguishes are supplied here rather than
      // spelled at the call site; the projection is safe because the pair
      // does not compose with anything that would misread the scale.
      auto magnitude =
          ondsp::FixedAttr::get(&context, ondsp::Signedness::Unsigned, elementType, 16);
      auto exponent = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, 11);
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
          ondsp::FpAttr::get(&context, elementType, *kernel.fpContract), ondsp::RoundingModeAttr());
    } else if (kernel.ast.result.kind == ReductionKind::Sine) {
      result = builder.create<ir::SineOp>(expressionLocation, outputType, lhs, numeric, rounding);
    } else if (kernel.ast.result.kind == ReductionKind::Cosine) {
      result = builder.create<ir::CosineOp>(expressionLocation, outputType, lhs, numeric, rounding);
    } else {
      result = builder.create<ir::RmsOp>(
          expressionLocation, outputType, lhs,
          kernel.fpContract
              ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
              : Attribute(numeric),
          kernel.fpContract ? ondsp::RoundingModeAttr() : rounding);
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
  if (kernel.ast.result.kind == ReductionKind::SosDf2Fixed) {
    Value scales = arguments.lookup(*getParameterOperand(kernel.ast.result, 2));
    Value state = arguments.lookup(*getParameterOperand(kernel.ast.result, 3));
    auto numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, 15);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto accumulatorType =
        ondsp::AccType::get(&context, builder.getIntegerType(kernel.ast.result.accumulatorWidth),
                            30, ondsp::Signedness::Signed, *kernel.updateOverflow);
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
    auto fixed = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, 15);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto accumulatorType =
        ondsp::AccType::get(&context, builder.getIntegerType(kernel.ast.result.accumulatorWidth),
                            30, ondsp::Signedness::Signed, *kernel.updateOverflow);
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
        kernel.fpContract ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
                          : Attribute(ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed,
                                                            elementType, 15));
    ondsp::RoundingModeAttr rounding;
    if (!kernel.fpContract)
      rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
    auto product = builder.create<ir::MatmulOp>(
        expressionLocation, cast<RankedTensorType>(resultType), lhs, rhs, numeric, rounding);
    builder.create<func::ReturnOp>(expressionLocation, product.getResult());
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }
  if (kernel.ast.result.kind == ReductionKind::Lms) {
    Value weights = arguments.lookup(*getParameterOperand(kernel.ast.result, 2));
    Attribute numeric =
        kernel.fpContract ? Attribute(ondsp::FpAttr::get(&context, elementType, *kernel.fpContract))
                          : Attribute(ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed,
                                                            elementType, 15));
    ondsp::RoundingModeAttr rounding;
    if (!kernel.fpContract)
      rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
    auto lms = builder.create<ir::LmsOp>(
        expressionLocation, resultTypes[0], resultTypes[1], lhs, rhs, weights,
        kernel.fpContract ? IntegerAttr() : builder.getI64IntegerAttr(kernel.ast.result.stepSize),
        kernel.fpContract ? builder.getF32FloatAttr(kernel.ast.result.fpConstant) : FloatAttr(),
        numeric, rounding);
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
    unsigned accumulatorFractionalBits = fractionalBits * 2;
    auto numeric =
        ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, fractionalBits);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto accumulatorType = ondsp::AccType::get(&context, builder.getIntegerType(accumulatorWidth),
                                               accumulatorFractionalBits, ondsp::Signedness::Signed,
                                               *kernel.updateOverflow);
    Value accumulator;
    if (kernel.ast.result.kind == ReductionKind::Dot)
      accumulator =
          builder.create<ir::DotOp>(expressionLocation, accumulatorType, lhs, rhs, numeric, product)
              .getResult();
    else
      accumulator =
          builder.create<ir::FirOp>(expressionLocation, accumulatorType, lhs, rhs, numeric, product)
              .getResult();
    auto result =
        builder.create<ondsp::AccExportOp>(expressionLocation, elementType, accumulator, numeric,
                                           *kernel.rounding, *kernel.destinationOverflow);
    builder.create<func::ReturnOp>(expressionLocation, result.getResult());
  } else {
    auto numeric = ondsp::FpAttr::get(&context, elementType, *kernel.fpContract);
    Value result;
    if (kernel.ast.result.kind == ReductionKind::Dot)
      result = builder.create<ir::DotOp>(expressionLocation, elementType, lhs, rhs, numeric,
                                         ondsp::ProductAttr());
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
