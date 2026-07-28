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
  Dct,
  MovingAverage,
  Gain,
  Rms
};

static bool isCfftKind(ReductionKind kind) {
  return kind == ReductionKind::Cfft || kind == ReductionKind::Icfft;
}

static bool isFftKind(ReductionKind kind) {
  return isCfftKind(kind) || kind == ReductionKind::Rfft || kind == ReductionKind::Irfft;
}

static bool isFftComposableKind(ReductionKind kind) {
  return isFftKind(kind) || kind == ReductionKind::Magnitude;
}

static bool isUnaryTensorKind(ReductionKind kind) {
  return kind == ReductionKind::Dct || kind == ReductionKind::MovingAverage ||
         kind == ReductionKind::Gain || kind == ReductionKind::Rms;
}

static bool isUnaryKind(ReductionKind kind) {
  return isFftComposableKind(kind) || isUnaryTensorKind(kind);
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
  ExpressionAst(const ExpressionAst &) = delete;
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
  SourcePosition position;
};

ExpressionAst::ExpressionAst(BuiltinCallAst call)
    : call(std::make_unique<BuiltinCallAst>(std::move(call))), position(this->call->position) {}
ExpressionAst::~ExpressionAst() = default;
ExpressionAst::ExpressionAst(ExpressionAst &&) noexcept = default;
ExpressionAst &ExpressionAst::operator=(ExpressionAst &&) noexcept = default;

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

  std::optional<KernelAst> parse() {
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
    if (!expect(TokenKind::Colon, "expected ':' before function body") ||
        !expectIdentifier("return", "expected a single return statement"))
      return std::nullopt;

    if (!isIdentifier("dot") && !isIdentifier("fir") && !isIdentifier("fir_filter") &&
        !isIdentifier("fir_decimate") && !isIdentifier("fir_interpolate") &&
        !isIdentifier("fir_stream") && !isIdentifier("sos_df2_fixed") &&
        !isIdentifier("convolution") && !isIdentifier("correlation") &&
        !isIdentifier("butterfly") && !isIdentifier("cfft") && !isIdentifier("icfft") &&
        !isIdentifier("rfft") && !isIdentifier("irfft") && !isIdentifier("magnitude") &&
        !isIdentifier("dct") && !isIdentifier("moving_average") && !isIdentifier("gain") &&
        !isIdentifier("rms")) {
      diagnostics.error(current.position,
                        "expected dot(...), fir(...), fir_filter(...), fir_decimate(...), "
                        "fir_interpolate(...), fir_stream(...), sos_df2_fixed(...), "
                        "convolution(...), correlation(...), butterfly(...), cfft(...), or "
                        "icfft(...), rfft(...), irfft(...), magnitude(...), dct(...), "
                        "moving_average(...), gain(...), or rms(...) return expression");
      return std::nullopt;
    }
    if (isIdentifier("dot"))
      kernel.result.kind = ReductionKind::Dot;
    else if (isIdentifier("fir"))
      kernel.result.kind = ReductionKind::Fir;
    else if (isIdentifier("fir_filter"))
      kernel.result.kind = ReductionKind::FirFilter;
    else if (isIdentifier("fir_decimate"))
      kernel.result.kind = ReductionKind::FirDecimate;
    else if (isIdentifier("fir_interpolate"))
      kernel.result.kind = ReductionKind::FirInterpolate;
    else if (isIdentifier("fir_stream"))
      kernel.result.kind = ReductionKind::FirStream;
    else if (isIdentifier("sos_df2_fixed"))
      kernel.result.kind = ReductionKind::SosDf2Fixed;
    else if (isIdentifier("convolution"))
      kernel.result.kind = ReductionKind::Convolution;
    else if (isIdentifier("correlation"))
      kernel.result.kind = ReductionKind::Correlation;
    else if (isIdentifier("butterfly"))
      kernel.result.kind = ReductionKind::Butterfly;
    else if (isIdentifier("cfft"))
      kernel.result.kind = ReductionKind::Cfft;
    else if (isIdentifier("icfft"))
      kernel.result.kind = ReductionKind::Icfft;
    else if (isIdentifier("rfft"))
      kernel.result.kind = ReductionKind::Rfft;
    else if (isIdentifier("irfft"))
      kernel.result.kind = ReductionKind::Irfft;
    else if (isIdentifier("magnitude"))
      kernel.result.kind = ReductionKind::Magnitude;
    else if (isIdentifier("dct"))
      kernel.result.kind = ReductionKind::Dct;
    else if (isIdentifier("moving_average"))
      kernel.result.kind = ReductionKind::MovingAverage;
    else if (isIdentifier("gain"))
      kernel.result.kind = ReductionKind::Gain;
    else
      kernel.result.kind = ReductionKind::Rms;
    kernel.result.position = current.position;
    advance();
    if (!expect(TokenKind::LeftParen, "expected '(' after builtin"))
      return std::nullopt;
    if (isFftKind(kernel.result.kind)) {
      std::optional<ExpressionAst> operand = parseFftExpression();
      if (!operand)
        return std::nullopt;
      kernel.result.operands.push_back(std::move(*operand));
      if (!expect(TokenKind::RightParen, "expected ')' after FFT operand"))
        return std::nullopt;
      if (current.kind != TokenKind::Eof) {
        diagnostics.error(current.position, "only one kernel is supported per file in this slice");
        return std::nullopt;
      }
      return kernel;
    }
    if (kernel.result.kind == ReductionKind::Magnitude) {
      std::optional<ExpressionAst> operand = parseFftExpression();
      if (!operand)
        return std::nullopt;
      kernel.result.operands.push_back(std::move(*operand));
      if (!expect(TokenKind::RightParen, "expected ')' after magnitude operand"))
        return std::nullopt;
      if (current.kind != TokenKind::Eof) {
        diagnostics.error(current.position, "only one kernel is supported per file in this slice");
        return std::nullopt;
      }
      return kernel;
    }
    auto lhs = parseIdentifier("expected builtin operand");
    if (!lhs)
      return std::nullopt;
    kernel.result.operands.emplace_back(lhs->spelling.str(), lhs->position);
    if (kernel.result.kind == ReductionKind::SosDf2Fixed) {
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
      if (!state || !expect(TokenKind::Comma, "expected ',' before sos_df2_fixed numeric policy"))
        return std::nullopt;
      kernel.result.operands.emplace_back(coefficients->spelling.str(), coefficients->position);
      kernel.result.operands.emplace_back(scales->spelling.str(), scales->position);
      kernel.result.operands.emplace_back(state->spelling.str(), state->position);
      if (!parseFixedSosPolicy(kernel.result) ||
          !expect(TokenKind::RightParen, "expected ')' after sos_df2_fixed expression"))
        return std::nullopt;
      if (current.kind != TokenKind::Eof) {
        diagnostics.error(current.position, "only one kernel is supported per file in this slice");
        return std::nullopt;
      }
      return kernel;
    }
    if (kernel.result.kind == ReductionKind::Butterfly ||
        kernel.result.kind == ReductionKind::FirStream) {
      llvm::StringRef builtin =
          kernel.result.kind == ReductionKind::Butterfly ? "butterfly" : "fir_stream";
      if (!expect(TokenKind::Comma,
                  llvm::Twine("expected ',' after ") + builtin + " first operand"))
        return std::nullopt;
      auto rhs = parseIdentifier(llvm::Twine("expected ") + builtin + " second operand");
      if (!rhs || !expect(TokenKind::Comma,
                          llvm::Twine("expected ',' after ") + builtin + " second operand"))
        return std::nullopt;
      auto third = parseIdentifier(kernel.result.kind == ReductionKind::Butterfly
                                       ? "expected butterfly twiddle operand"
                                       : "expected fir_stream state operand");
      if (!third || !expect(TokenKind::RightParen,
                            llvm::Twine("expected ')' after ") + builtin + " operands"))
        return std::nullopt;
      kernel.result.operands.emplace_back(rhs->spelling.str(), rhs->position);
      kernel.result.operands.emplace_back(third->spelling.str(), third->position);
      if (kernel.result.kind == ReductionKind::FirStream) {
        kernel.result.accumulatorAuto = true;
        kernel.result.rounding = "nearest_even";
        kernel.result.destinationOverflow = "saturate";
        kernel.result.updateOverflow = "wrap";
      }
      if (current.kind != TokenKind::Eof) {
        diagnostics.error(current.position, "only one kernel is supported per file in this slice");
        return std::nullopt;
      }
      return kernel;
    }
    if (kernel.result.kind == ReductionKind::Dct || kernel.result.kind == ReductionKind::Rms) {
      llvm::StringRef builtin = kernel.result.kind == ReductionKind::Dct ? "dct" : "rms";
      if (!expect(TokenKind::RightParen, llvm::Twine("expected ')' after ") + builtin + " operand"))
        return std::nullopt;
      if (current.kind != TokenKind::Eof) {
        diagnostics.error(current.position, "only one kernel is supported per file in this slice");
        return std::nullopt;
      }
      return kernel;
    }
    if (kernel.result.kind == ReductionKind::MovingAverage ||
        kernel.result.kind == ReductionKind::Gain) {
      bool isGain = kernel.result.kind == ReductionKind::Gain;
      llvm::StringRef builtin = isGain ? "gain" : "moving_average";
      llvm::StringRef keyword = isGain ? "gain" : "window";
      if (!expect(TokenKind::Comma, llvm::Twine("expected ',' before ") + builtin + " constant") ||
          !expectIdentifier(keyword, llvm::Twine("expected ") + builtin + " constant") ||
          !expect(TokenKind::Equal, llvm::Twine("expected '=' after ") + keyword))
        return std::nullopt;
      auto constant = parseSignedInteger(llvm::Twine("expected ") + builtin + " constant");
      if (!constant)
        return std::nullopt;
      if (isGain)
        kernel.result.gain = *constant;
      else
        kernel.result.window = *constant;
      if (!expect(TokenKind::RightParen,
                  llvm::Twine("expected ')' after ") + builtin + " expression"))
        return std::nullopt;
      if (current.kind != TokenKind::Eof) {
        diagnostics.error(current.position, "only one kernel is supported per file in this slice");
        return std::nullopt;
      }
      return kernel;
    }
    if (!expect(TokenKind::Comma, "expected ',' after reduction left operand"))
      return std::nullopt;
    auto rhs = parseIdentifier("expected reduction right operand");
    if (!rhs)
      return std::nullopt;
    kernel.result.operands.emplace_back(rhs->spelling.str(), rhs->position);

    if (kernel.result.kind == ReductionKind::FirFilter) {
      if (!expect(TokenKind::Comma, "expected ',' before FIR boundary policy"))
        return std::nullopt;
      if (!expectIdentifier("boundary", "expected FIR boundary policy") ||
          !expect(TokenKind::Equal, "expected '=' after boundary"))
        return std::nullopt;
      auto boundary = parseIdentifier("expected FIR boundary mode");
      if (!boundary || !expect(TokenKind::Comma, "expected ',' after FIR boundary mode"))
        return std::nullopt;
      kernel.result.boundary = boundary->spelling.str();
    } else if (kernel.result.kind == ReductionKind::FirDecimate ||
               kernel.result.kind == ReductionKind::FirInterpolate) {
      bool isInterpolation = kernel.result.kind == ReductionKind::FirInterpolate;
      llvm::StringRef operation = isInterpolation ? "fir_interpolate" : "fir_decimate";
      if (!expect(TokenKind::Comma, "expected ',' before FIR resampling factor") ||
          !expectIdentifier("factor", "expected FIR resampling factor") ||
          !expect(TokenKind::Equal, "expected '=' after factor"))
        return std::nullopt;
      auto factor = parseSignedInteger("expected FIR resampling factor");
      if (!factor)
        return std::nullopt;
      kernel.result.factor = *factor;
      kernel.result.accumulatorAuto = true;
      kernel.result.rounding = "nearest_even";
      kernel.result.destinationOverflow = "saturate";
      kernel.result.updateOverflow = "wrap";
      if (!expect(TokenKind::RightParen,
                  llvm::Twine("expected ')' after ") + operation + " expression"))
        return std::nullopt;
      if (current.kind != TokenKind::Eof) {
        diagnostics.error(current.position, "only one kernel is supported per file in this slice");
        return std::nullopt;
      }
      return kernel;
    } else if (current.kind == TokenKind::RightParen &&
               kernel.primaryResult().type != SourceType::F32) {
      kernel.result.accumulatorAuto = true;
      kernel.result.rounding = "nearest_even";
      kernel.result.destinationOverflow = "saturate";
      kernel.result.updateOverflow = "wrap";
    } else if (!expect(TokenKind::Comma, "expected ',' before numeric policy")) {
      return std::nullopt;
    }

    if (kernel.primaryResult().type != SourceType::F32) {
      if (!kernel.result.accumulatorAuto && !parseFixedPolicy(kernel.result))
        return std::nullopt;
    } else {
      if (!expectIdentifier("contract", "expected floating-point contract policy") ||
          !expect(TokenKind::Equal, "expected '=' after contract"))
        return std::nullopt;
      auto contract = parseIdentifier("expected floating-point contract mode");
      if (!contract)
        return std::nullopt;
      kernel.result.fpContract = contract->spelling.str();
    }

    if (!expect(TokenKind::RightParen, "expected ')' after reduction expression"))
      return std::nullopt;

    if (current.kind != TokenKind::Eof) {
      diagnostics.error(current.position, "only one kernel is supported per file in this slice");
      return std::nullopt;
    }
    return kernel;
  }

private:
  std::optional<ExpressionAst> parseFftExpression() {
    bool isFftCall = isIdentifier("cfft") || isIdentifier("icfft") || isIdentifier("rfft") ||
                     isIdentifier("irfft");
    if (!isFftCall || next.kind != TokenKind::LeftParen) {
      auto parameter = parseIdentifier("expected FFT operand expression");
      if (!parameter)
        return std::nullopt;
      return ExpressionAst(parameter->spelling.str(), parameter->position);
    }

    BuiltinCallAst call;
    call.position = current.position;
    if (isIdentifier("cfft"))
      call.kind = ReductionKind::Cfft;
    else if (isIdentifier("icfft"))
      call.kind = ReductionKind::Icfft;
    else if (isIdentifier("rfft"))
      call.kind = ReductionKind::Rfft;
    else
      call.kind = ReductionKind::Irfft;
    advance();
    if (!expect(TokenKind::LeftParen, "expected '(' after nested FFT builtin"))
      return std::nullopt;
    std::optional<ExpressionAst> operand = parseFftExpression();
    if (!operand)
      return std::nullopt;
    call.operands.push_back(std::move(*operand));
    if (!expect(TokenKind::RightParen, "expected ')' after nested FFT operand"))
      return std::nullopt;
    return ExpressionAst(std::move(call));
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

  Lexer &lexer;
  Diagnostics &diagnostics;
  Token current;
  Token next;
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
  return std::nullopt;
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
  bool hasThreeOperands =
      ast.result.kind == ReductionKind::Butterfly || ast.result.kind == ReductionKind::FirStream;
  bool hasFourOperands = ast.result.kind == ReductionKind::SosDf2Fixed;
  size_t expectedOperandCount = isUnaryKind(ast.result.kind) ? 1
                                : hasFourOperands            ? 4
                                : hasThreeOperands           ? 3
                                                             : 2;
  if (ast.result.operands.size() != expectedOperandCount) {
    diagnostics.error(ast.result.position, "builtin operand count does not match its contract");
    return std::nullopt;
  }
  if (!isFftComposableKind(ast.result.kind) &&
      llvm::any_of(ast.result.operands,
                   [](const ExpressionAst &operand) { return !operand.isParameterReference(); })) {
    diagnostics.error(ast.result.position,
                      "nested calls are currently supported only by FFT-family builtins");
    return std::nullopt;
  }
  size_t expectedParameterCount = isUnaryKind(ast.result.kind) ? 1
                                  : hasFourOperands            ? 4
                                  : hasThreeOperands           ? 3
                                                               : 2;
  if (ast.parameters.size() != expectedParameterCount) {
    diagnostics.error(ast.position,
                      isFftKind(ast.result.kind) ? "FFT kernels require exactly one parameter"
                      : isUnaryKind(ast.result.kind)
                          ? "unary DSP kernels require exactly one parameter"
                      : hasFourOperands  ? "sos_df2_fixed kernels require exactly four parameters"
                      : hasThreeOperands ? "butterfly and fir_stream kernels require exactly three "
                                           "parameters"
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
    if (!isFftComposableKind(ast.result.kind) && parameter.type != ast.primaryResult().type) {
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
  if (!isFftComposableKind(ast.result.kind) && (!lhsName || !parameterNames.contains(*lhsName))) {
    diagnostics.error(ast.result.position, llvm::Twine("unknown builtin operand '") +
                                               (lhsName ? *lhsName : "<nested call>") + "'");
    return std::nullopt;
  }
  if (!isUnaryKind(ast.result.kind) && (!rhsName || !parameterNames.contains(*rhsName))) {
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
    diagnostics.error(ast.result.position, "unknown sos_df2_fixed builtin operand");
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
    return CheckedKernel{std::move(ast), *updateOverflow, *outputRounding, *outputOverflow,
                         std::nullopt,   *stateRounding,  *stateOverflow};
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
  if (isFftComposableKind(ast.result.kind)) {
    struct FftExpressionType {
      SourceType elementType;
      int64_t extent;
    };
    std::function<std::optional<FftExpressionType>(const ExpressionAst &)> checkFftExpression;
    std::function<std::optional<FftExpressionType>(const BuiltinCallAst &)> checkFftCall;
    checkFftExpression = [&](const ExpressionAst &expression) -> std::optional<FftExpressionType> {
      if (expression.isParameterReference()) {
        auto parameter = parametersByName.find(expression.parameter);
        if (parameter == parametersByName.end()) {
          diagnostics.error(expression.position,
                            llvm::Twine("unknown FFT operand '") + expression.parameter + "'");
          return std::nullopt;
        }
        const ParameterAst *value = parameter->second;
        if (!value->isTensor() || !hasRank(value->shape, 1)) {
          diagnostics.error(expression.position,
                            "FFT-family builtins currently require rank-1 tensor operands");
          return std::nullopt;
        }
        const std::optional<int64_t> &extent = getRankOneExtent(value->shape);
        if (!extent) {
          diagnostics.error(expression.position,
                            "FFT-family builtins currently require static operand extents");
          return std::nullopt;
        }
        return FftExpressionType{value->type, *extent};
      }

      return checkFftCall(*expression.call);
    };
    checkFftCall = [&](const BuiltinCallAst &call) -> std::optional<FftExpressionType> {
      if (!isFftComposableKind(call.kind) || call.operands.size() != 1) {
        diagnostics.error(call.position,
                          "nested calls are currently supported only by unary FFT-family builtins");
        return std::nullopt;
      }
      std::optional<FftExpressionType> input = checkFftExpression(call.operands.front());
      if (!input)
        return std::nullopt;

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
        return FftExpressionType{SourceType::Q15, input->extent};
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
        if (input->extent != 8 && input->extent != 16) {
          diagnostics.error(call.position, "rfft currently supports only eight or sixteen points");
          return std::nullopt;
        }
        return FftExpressionType{SourceType::ComplexQ15, input->extent / 2 + 1};
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
      return FftExpressionType{SourceType::Q15, (input->extent - 1) * 2};
    };

    std::optional<FftExpressionType> inferred = checkFftCall(ast.result);
    if (!inferred)
      return std::nullopt;
    if (!ast.primaryResult().tensor || !hasRank(ast.primaryResult().shape, 1)) {
      diagnostics.error(ast.result.position,
                        "FFT-family builtins currently require a rank-1 tensor result");
      return std::nullopt;
    }
    const std::optional<int64_t> &resultExtent = getRankOneExtent(ast.primaryResult().shape);
    if (!resultExtent) {
      diagnostics.error(ast.result.position,
                        "FFT-family builtins currently require a static result extent");
      return std::nullopt;
    }
    if (ast.primaryResult().type != inferred->elementType || *resultExtent != inferred->extent) {
      diagnostics.error(ast.result.position,
                        "declared FFT result type does not match the builtin expression");
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
        ast.primaryResult().type != SourceType::Q15 || !lhsParameter || !rhsParameter ||
        !lhsParameter->isTensor() || !rhsParameter->isTensor()) {
      diagnostics.error(ast.result.position,
                        "fir_decimate requires Q15 tensor input, coefficients, and result");
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
        ast.primaryResult().type != SourceType::Q15 || !lhsParameter || !rhsParameter ||
        !lhsParameter->isTensor() || !rhsParameter->isTensor()) {
      diagnostics.error(ast.result.position,
                        "fir_interpolate requires Q15 tensor input, coefficients, and result");
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
                                                                                : "rms";
    if (constexprCount != 0 || !ast.primaryResult().tensor ||
        ast.primaryResult().type != SourceType::Q15 || !lhsParameter || !lhsParameter->isTensor()) {
      diagnostics.error(ast.result.position,
                        llvm::Twine(builtin) + " requires a Q15 tensor input and result");
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
      if (window < 2 || window > 64 || !llvm::isPowerOf2_64(window)) {
        diagnostics.error(ast.result.position,
                          "moving_average currently requires a power-of-two window in [2, 64]");
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
      if (ast.result.gain < std::numeric_limits<int16_t>::min() ||
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
    } else {
      if (*inputExtent < 2 || *inputExtent > 4096 || !llvm::isPowerOf2_64(*inputExtent)) {
        diagnostics.error(ast.result.position,
                          "rms currently requires a power-of-two input extent in [2, 4096]");
        return std::nullopt;
      }
      if (*resultExtent != 1) {
        diagnostics.error(ast.result.position, "rms returns a single-element tensor");
        return std::nullopt;
      }
    }
    return CheckedKernel{std::move(ast), std::nullopt, ondsp::RoundingMode::NearestEven,
                         std::nullopt, std::nullopt};
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
    bool isWindowReduction = ast.result.kind == ReductionKind::Convolution ||
                             ast.result.kind == ReductionKind::Correlation ||
                             ast.result.kind == ReductionKind::FirDecimate ||
                             ast.result.kind == ReductionKind::FirInterpolate;
    if (ast.primaryResult().type != SourceType::Q15 || (!isScalarReduction && !isWindowReduction)) {
      diagnostics.error(
          ast.result.position,
          "automatic accumulation currently supports static Q15 dot, fir, fir_decimate, "
          "fir_interpolate, and conv1d");
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
  if (isFftComposableKind(kernel.ast.result.kind)) {
    auto layout = ondsp::CxLayoutAttr::get(&context, ondsp::ComplexLayout::PackedI16ImagHiRealLo);
    auto i16 = builder.getI16Type();
    auto numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, i16, 15);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto productScale = ondsp::ScaleAttr::get(&context, 0, 15, ondsp::RoundingMode::NearestEven,
                                              ondsp::OverflowMode::Saturate, i16);
    auto outputScale = ondsp::ScaleAttr::get(&context, 0, 1, ondsp::RoundingMode::NearestEven,
                                             ondsp::OverflowMode::Saturate, i16);
    std::function<Value(const BuiltinCallAst &)> emitFftCall =
        [&](const BuiltinCallAst &call) -> Value {
      const ExpressionAst &operand = call.operands.front();
      Value input = operand.isParameterReference() ? arguments.lookup(operand.parameter)
                                                   : emitFftCall(*operand.call);
      auto inputType = cast<RankedTensorType>(input.getType());
      Location callLocation = getLocation(context, sourceName, call.position);
      if (call.kind == ReductionKind::Magnitude) {
        auto outputType = RankedTensorType::get({inputType.getDimSize(0)}, builder.getI16Type());
        return builder.create<ir::CxMagnitudeOp>(
            callLocation, outputType, input, layout, numeric,
            ondsp::RoundingModeAttr::get(&context, ondsp::RoundingMode::NearestEven));
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
    Value result = emitFftCall(kernel.ast.result);
    builder.create<func::ReturnOp>(expressionLocation, result);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }

  Value lhs = arguments.lookup(*getParameterOperand(kernel.ast.result, 0));
  if (isUnaryTensorKind(kernel.ast.result.kind)) {
    auto outputType = cast<RankedTensorType>(resultType);
    auto numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, 15);
    auto rounding = ondsp::RoundingModeAttr::get(&context, *kernel.rounding);
    Value result;
    if (kernel.ast.result.kind == ReductionKind::Dct) {
      unsigned stageCount = llvm::Log2_64(outputType.getDimSize(0));
      auto outputNumeric =
          ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, 14 - stageCount);
      result =
          builder.create<ir::DctOp>(expressionLocation, outputType, lhs, numeric, outputNumeric);
    } else if (kernel.ast.result.kind == ReductionKind::MovingAverage) {
      result = builder.create<ir::MovingAverageOp>(
          expressionLocation, outputType, lhs, builder.getI64IntegerAttr(kernel.ast.result.window),
          numeric);
    } else if (kernel.ast.result.kind == ReductionKind::Gain) {
      result = builder.create<ir::GainOp>(expressionLocation, outputType, lhs,
                                          builder.getI64IntegerAttr(kernel.ast.result.gain),
                                          numeric, rounding);
    } else {
      result = builder.create<ir::RmsOp>(expressionLocation, outputType, lhs, numeric, rounding);
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

  bool isConv1D = kernel.ast.result.kind == ReductionKind::Convolution ||
                  kernel.ast.result.kind == ReductionKind::Correlation;
  bool isFirDecimate = kernel.ast.result.kind == ReductionKind::FirDecimate;
  bool isFirInterpolate = kernel.ast.result.kind == ReductionKind::FirInterpolate;
  if (kernel.ast.result.kind == ReductionKind::FirFilter || isFirDecimate || isFirInterpolate ||
      isConv1D) {
    auto outputType = cast<RankedTensorType>(resultType);
    if ((isFirDecimate || isFirInterpolate) && outputType.isDynamicDim(0)) {
      emitError(expressionLocation, "internal error: resampling source result must be static");
      return {};
    }
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
          builder.getI64IntegerAttr(kernel.ast.result.factor), cast<ondsp::FixedAttr>(numeric),
          product, accumulator, destination, rounding, overflow);
    } else if (isFirInterpolate) {
      result = builder.create<ir::FirInterpolateOp>(
          expressionLocation, outputType, lhs, rhs, init,
          builder.getI64IntegerAttr(kernel.ast.result.factor), cast<ondsp::FixedAttr>(numeric),
          product, accumulator, destination, rounding, overflow);
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

  std::optional<CheckedKernel> checked = checkKernel(std::move(*ast), diagnostics);
  if (!checked || diagnostics.failed())
    return {};
  return generateModule(*checked, sourceName, context);
}

} // namespace ondrix::frontend
