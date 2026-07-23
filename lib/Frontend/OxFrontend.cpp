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

#include <cctype>
#include <cstdint>
#include <limits>
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

enum class SourceType { Q15, Q31, F32 };

enum class ContainerKind { Buffer, Tensor, Constexpr };

enum class ReductionKind { Dot, Fir, FirFilter };

struct ParameterAst {
  std::string name;
  SourceType type;
  std::optional<int64_t> extent;
  ContainerKind container = ContainerKind::Buffer;
  std::vector<int64_t> constantValues;
  SourcePosition position;

  bool isBuffer() const { return container == ContainerKind::Buffer; }
  bool isTensor() const { return container == ContainerKind::Tensor; }
  bool isConstexpr() const { return container == ContainerKind::Constexpr; }
};

struct ReductionAst {
  ReductionKind kind;
  std::string lhs;
  std::string rhs;
  uint64_t accumulatorWidth = 0;
  std::string updateOverflow;
  std::string rounding;
  std::string destinationOverflow;
  std::string fpContract;
  std::string boundary;
  SourcePosition position;
};

struct KernelAst {
  std::string name;
  std::vector<ParameterAst> parameters;
  SourceType resultType;
  bool tensorResult = false;
  std::optional<int64_t> resultExtent;
  ReductionAst result;
  SourcePosition position;
};

class Parser {
public:
  Parser(Lexer &lexer, Diagnostics &diagnostics)
      : lexer(lexer), diagnostics(diagnostics), current(lexer.next()) {}

  std::optional<KernelAst> parse() {
    if (!isIdentifier("kernel")) {
      if (current.kind == TokenKind::Identifier)
        diagnostics.error(current.position, llvm::Twine("unsupported top-level construct '") +
                                                current.spelling + "'; expected 'kernel'");
      else
        diagnostics.error(current.position, "expected 'kernel'");
      return std::nullopt;
    }

    KernelAst kernel;
    kernel.position = current.position;
    advance();
    auto name = parseIdentifier("expected kernel name");
    if (!name)
      return std::nullopt;
    kernel.name = name->spelling.str();

    if (!expect(TokenKind::LeftParen, "expected '(' after kernel name"))
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
    if (isIdentifier("tensor")) {
      kernel.tensorResult = true;
      advance();
      if (!parseShapedType(kernel.resultType, kernel.resultExtent, "tensor"))
        return std::nullopt;
    } else {
      auto resultType = parseSourceType("expected kernel result type 'q15', 'q31', or 'f32'");
      if (!resultType)
        return std::nullopt;
      kernel.resultType = *resultType;
    }
    if (!expect(TokenKind::Colon, "expected ':' before kernel body") ||
        !expectIdentifier("return", "expected a single return statement"))
      return std::nullopt;

    if (!isIdentifier("dot") && !isIdentifier("fir") && !isIdentifier("fir_filter")) {
      diagnostics.error(current.position,
                        "expected dot(...), fir(...), or fir_filter(...) return expression");
      return std::nullopt;
    }
    if (isIdentifier("dot"))
      kernel.result.kind = ReductionKind::Dot;
    else if (isIdentifier("fir"))
      kernel.result.kind = ReductionKind::Fir;
    else
      kernel.result.kind = ReductionKind::FirFilter;
    kernel.result.position = current.position;
    advance();
    if (!expect(TokenKind::LeftParen, "expected '(' after reduction builtin"))
      return std::nullopt;
    auto lhs = parseIdentifier("expected reduction left operand");
    if (!lhs || !expect(TokenKind::Comma, "expected ',' after reduction left operand"))
      return std::nullopt;
    auto rhs = parseIdentifier("expected reduction right operand");
    if (!rhs || !expect(TokenKind::Comma, "expected ',' before numeric policy"))
      return std::nullopt;
    kernel.result.lhs = lhs->spelling.str();
    kernel.result.rhs = rhs->spelling.str();

    if (kernel.result.kind == ReductionKind::FirFilter) {
      if (!expectIdentifier("boundary", "expected FIR boundary policy") ||
          !expect(TokenKind::Equal, "expected '=' after boundary"))
        return std::nullopt;
      auto boundary = parseIdentifier("expected FIR boundary mode");
      if (!boundary || !expect(TokenKind::Comma, "expected ',' after FIR boundary mode"))
        return std::nullopt;
      kernel.result.boundary = boundary->spelling.str();
    }

    if (kernel.resultType != SourceType::F32) {
      if (!parseFixedPolicy(kernel.result))
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
  bool isIdentifier(llvm::StringRef spelling) const {
    return current.kind == TokenKind::Identifier && current.spelling == spelling;
  }

  void advance() { current = lexer.next(); }

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

  bool parseShapedType(SourceType &type, std::optional<int64_t> &extent,
                       llvm::StringRef containerName) {
    if (!expect(TokenKind::LeftBracket, llvm::Twine("expected '[' in ") + containerName + " type"))
      return false;
    auto parsedType = parseSourceType(llvm::Twine("expected ") + containerName +
                                      " element type 'q15', 'q31', or 'f32'");
    if (!parsedType)
      return false;
    type = *parsedType;
    if (current.kind == TokenKind::Comma) {
      advance();
      auto parsedExtent = parseInteger(llvm::Twine("expected static ") + containerName + " extent");
      uint64_t value = 0;
      if (!parsedExtent)
        return false;
      if (parsedExtent->spelling.getAsInteger(10, value) ||
          value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        diagnostics.error(parsedExtent->position,
                          llvm::Twine(containerName) + " extent is out of range");
        return false;
      }
      extent = static_cast<int64_t>(value);
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
      if (!parseShapedType(parameter.type, parameter.extent, containerName))
        return std::nullopt;
      return parameter;
    }

    if (!isIdentifier("constexpr")) {
      diagnostics.error(current.position,
                        "expected parameter type 'buffer[...]', 'tensor[...]', or "
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

  bool parseFixedPolicy(ReductionAst &result) {
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
        !expect(TokenKind::RightBracket, "expected ']' after accumulator policy") ||
        !expect(TokenKind::Comma, "expected ',' before rounding policy") ||
        !expectIdentifier("rounding", "expected rounding policy") ||
        !expect(TokenKind::Equal, "expected '=' after rounding"))
      return false;
    result.updateOverflow = updateOverflow->spelling.str();
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

  Lexer &lexer;
  Diagnostics &diagnostics;
  Token current;
};

struct CheckedKernel {
  KernelAst ast;
  std::optional<ondsp::OverflowMode> updateOverflow;
  std::optional<ondsp::RoundingMode> rounding;
  std::optional<ondsp::OverflowMode> destinationOverflow;
  std::optional<ondsp::FpContractMode> fpContract;
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

static std::optional<CheckedKernel> checkKernel(KernelAst ast, Diagnostics &diagnostics) {
  if (ast.parameters.size() != 2) {
    diagnostics.error(ast.position, "dot and FIR kernels require exactly two parameters");
    return std::nullopt;
  }
  if (ast.resultExtent && *ast.resultExtent <= 0) {
    diagnostics.error(ast.position, "static result tensor extent must be positive");
    return std::nullopt;
  }

  llvm::StringSet<> parameterNames;
  const ParameterAst *lhsParameter = nullptr;
  const ParameterAst *rhsParameter = nullptr;
  for (const ParameterAst &parameter : ast.parameters) {
    if (!parameterNames.insert(parameter.name).second) {
      diagnostics.error(parameter.position,
                        llvm::Twine("duplicate parameter '") + parameter.name + "'");
      return std::nullopt;
    }
    if (parameter.type != ast.resultType) {
      diagnostics.error(parameter.position,
                        "parameter element types must match the kernel result type");
      return std::nullopt;
    }
    if (parameter.extent && *parameter.extent <= 0) {
      diagnostics.error(parameter.position, "static shaped extent must be positive");
      return std::nullopt;
    }
    if (parameter.name == ast.result.lhs)
      lhsParameter = &parameter;
    if (parameter.name == ast.result.rhs)
      rhsParameter = &parameter;
  }
  if (!parameterNames.contains(ast.result.lhs)) {
    diagnostics.error(ast.result.position,
                      llvm::Twine("unknown reduction operand '") + ast.result.lhs + "'");
    return std::nullopt;
  }
  if (!parameterNames.contains(ast.result.rhs)) {
    diagnostics.error(ast.result.position,
                      llvm::Twine("unknown reduction operand '") + ast.result.rhs + "'");
    return std::nullopt;
  }

  if (ast.result.kind == ReductionKind::FirFilter) {
    if (!ast.tensorResult) {
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
    if (ast.result.boundary != "valid") {
      diagnostics.error(ast.result.position, "fir_filter currently supports only boundary=valid");
      return std::nullopt;
    }
    std::optional<int64_t> coefficientExtent = rhsParameter->extent;
    if (rhsParameter->isConstexpr())
      coefficientExtent = static_cast<int64_t>(rhsParameter->constantValues.size());
    if (lhsParameter->extent && coefficientExtent) {
      if (*coefficientExtent > *lhsParameter->extent) {
        diagnostics.error(ast.result.position,
                          "valid fir_filter requires input extent at least coefficient extent");
        return std::nullopt;
      }
      int64_t expectedExtent = *lhsParameter->extent - *coefficientExtent + 1;
      if (ast.resultExtent && *ast.resultExtent != expectedExtent) {
        diagnostics.error(ast.result.position,
                          "static fir_filter result extent does not match valid convolution");
        return std::nullopt;
      }
    }
  } else {
    if (ast.tensorResult) {
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

  if (ast.resultType == SourceType::F32) {
    if (llvm::any_of(ast.parameters,
                     [](const ParameterAst &parameter) { return parameter.isConstexpr(); })) {
      diagnostics.error(ast.result.position,
                        "constexpr parameters are restricted to fixed-point FIR coefficients");
      return std::nullopt;
    }
    auto contract = parseFpContract(ast.result.fpContract);
    if (!contract) {
      diagnostics.error(ast.result.position, llvm::Twine("unsupported floating-point contract '") +
                                                 ast.result.fpContract + "'");
      return std::nullopt;
    }
    return CheckedKernel{std::move(ast), std::nullopt, std::nullopt, std::nullopt, *contract};
  }

  unsigned constexprCount = llvm::count_if(
      ast.parameters, [](const ParameterAst &parameter) { return parameter.isConstexpr(); });
  if (constexprCount != 0) {
    if (constexprCount != 1 || !rhsParameter || !rhsParameter->isConstexpr() || !lhsParameter ||
        lhsParameter->isConstexpr()) {
      diagnostics.error(
          ast.result.position,
          "constexpr is supported only for the right operand of a fixed-point reduction");
      return std::nullopt;
    }
    if (rhsParameter->constantValues.empty()) {
      diagnostics.error(rhsParameter->position, "constexpr reduction operand cannot be empty");
      return std::nullopt;
    }
    int64_t minimum = ast.resultType == SourceType::Q15
                          ? static_cast<int64_t>(std::numeric_limits<int16_t>::min())
                          : static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    int64_t maximum = ast.resultType == SourceType::Q15
                          ? static_cast<int64_t>(std::numeric_limits<int16_t>::max())
                          : static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    for (int64_t value : rhsParameter->constantValues) {
      if (value < minimum || value > maximum) {
        diagnostics.error(rhsParameter->position,
                          ast.resultType == SourceType::Q15
                              ? "Q15 constexpr coefficient is outside signed i16 storage range"
                              : "Q31 constexpr coefficient is outside signed i32 storage range");
        return std::nullopt;
      }
    }
    if (ast.result.kind != ReductionKind::FirFilter) {
      if (!lhsParameter->extent) {
        diagnostics.error(lhsParameter->position,
                          "a constexpr reduction operand requires a static left operand extent");
        return std::nullopt;
      }
      if (*lhsParameter->extent != static_cast<int64_t>(rhsParameter->constantValues.size())) {
        diagnostics.error(lhsParameter->position,
                          "static input extent must equal the constexpr coefficient count");
        return std::nullopt;
      }
    }
  }

  uint64_t requiredAccumulatorWidth = ast.resultType == SourceType::Q15 ? 40 : 64;
  if (ast.result.accumulatorWidth != requiredAccumulatorWidth) {
    diagnostics.error(ast.result.position,
                      ast.resultType == SourceType::Q15
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

  Type elementType;
  if (kernel.ast.resultType == SourceType::Q15)
    elementType = builder.getI16Type();
  else if (kernel.ast.resultType == SourceType::Q31)
    elementType = builder.getI32Type();
  else
    elementType = builder.getF32Type();
  SmallVector<Type> inputTypes;
  for (const ParameterAst &parameter : kernel.ast.parameters) {
    if (parameter.isConstexpr())
      continue;
    int64_t extent = parameter.extent.value_or(ShapedType::kDynamic);
    if (parameter.isTensor())
      inputTypes.push_back(RankedTensorType::get({extent}, elementType));
    else
      inputTypes.push_back(MemRefType::get({extent}, elementType));
  }
  Type resultType = elementType;
  if (kernel.ast.tensorResult) {
    int64_t extent = kernel.ast.resultExtent.value_or(ShapedType::kDynamic);
    resultType = RankedTensorType::get({extent}, elementType);
  }
  FunctionType functionType = builder.getFunctionType(inputTypes, {resultType});
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
    MemRefType coefficientType = MemRefType::get({extent}, elementType);
    RankedTensorType initializerType = RankedTensorType::get({extent}, elementType);
    SmallVector<llvm::APInt> values;
    values.reserve(parameter.constantValues.size());
    unsigned storageWidth = cast<IntegerType>(elementType).getWidth();
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
  Value lhs = arguments.lookup(kernel.ast.result.lhs);
  Value rhs = arguments.lookup(kernel.ast.result.rhs);
  if (kernel.ast.result.kind == ReductionKind::FirFilter) {
    auto outputType = cast<RankedTensorType>(resultType);
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
    if (kernel.ast.resultType == SourceType::F32) {
      numeric = ondsp::FpAttr::get(&context, elementType, *kernel.fpContract);
    } else {
      unsigned storageWidth = cast<IntegerType>(elementType).getWidth();
      unsigned fractionalBits = storageWidth - 1;
      unsigned accumulatorWidth = kernel.ast.resultType == SourceType::Q15 ? 40 : 64;
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

    Value result = builder.create<ir::FirFilterOp>(
        expressionLocation, outputType, lhs, rhs, init, Value(), ir::FirBoundaryMode::Valid,
        numeric, product, accumulator, destination, rounding, overflow);
    builder.create<func::ReturnOp>(expressionLocation, result);
    module->push_back(function);
    if (failed(verify(*module)))
      return {};
    return module;
  }

  if (kernel.ast.resultType != SourceType::F32) {
    unsigned storageWidth = cast<IntegerType>(elementType).getWidth();
    unsigned fractionalBits = storageWidth - 1;
    unsigned accumulatorWidth = kernel.ast.resultType == SourceType::Q15 ? 40 : 64;
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
