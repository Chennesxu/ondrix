#include "ondrix/Frontend/OxFrontend.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <cstdint>
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
    if (current == '-' && offset + 1 < source.size() && source[offset + 1] == '>') {
      advance();
      advance();
      return {TokenKind::Arrow, "->", start};
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

enum class SourceType { Q15, F32 };

enum class ReductionKind { Dot, Fir };

struct ParameterAst {
  std::string name;
  SourceType type;
  SourcePosition position;
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
  SourcePosition position;
};

struct KernelAst {
  std::string name;
  std::vector<ParameterAst> parameters;
  SourceType resultType;
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
      auto parameterName = parseIdentifier("expected parameter name");
      if (!parameterName)
        return std::nullopt;
      if (!expect(TokenKind::Colon, "expected ':' after parameter name") ||
          !expectIdentifier("buffer", "expected parameter type 'buffer[q15]' or 'buffer[f32]'") ||
          !expect(TokenKind::LeftBracket, "expected '[' in buffer type"))
        return std::nullopt;
      auto parameterType = parseSourceType("expected buffer element type 'q15' or 'f32'");
      if (!parameterType || !expect(TokenKind::RightBracket, "expected ']' in buffer type"))
        return std::nullopt;
      kernel.parameters.push_back(
          {parameterName->spelling.str(), *parameterType, parameterName->position});
      if (current.kind != TokenKind::Comma)
        break;
      advance();
    } while (true);

    if (!expect(TokenKind::RightParen, "expected ')' after parameters") ||
        !expect(TokenKind::Arrow, "expected '->' after parameters"))
      return std::nullopt;
    auto resultType = parseSourceType("expected kernel result type 'q15' or 'f32'");
    if (!resultType || !expect(TokenKind::Colon, "expected ':' before kernel body") ||
        !expectIdentifier("return", "expected a single return statement"))
      return std::nullopt;
    kernel.resultType = *resultType;

    if (!isIdentifier("dot") && !isIdentifier("fir")) {
      diagnostics.error(current.position, "expected dot(...) or fir(...) return expression");
      return std::nullopt;
    }
    kernel.result.kind = isIdentifier("dot") ? ReductionKind::Dot : ReductionKind::Fir;
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

    if (kernel.resultType == SourceType::Q15) {
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
    if (isIdentifier("f32")) {
      advance();
      return SourceType::F32;
    }
    diagnostics.error(current.position, message);
    return std::nullopt;
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

  llvm::StringSet<> parameterNames;
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

  if (ast.resultType == SourceType::F32) {
    if (ast.result.kind != ReductionKind::Dot) {
      diagnostics.error(ast.result.position,
                        "the current f32 frontend slice supports dot but not FIR");
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

  if (ast.result.accumulatorWidth != 40) {
    diagnostics.error(ast.result.position,
                      "the executable Q15 profile requires exact accumulator width 40");
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
  context.loadDialect<func::FuncDialect, ir::OndrixDialect, ondsp::OndspDialect>();
  OpBuilder builder(&context);
  Location kernelLocation = getLocation(context, sourceName, kernel.ast.position);
  OwningOpRef<ModuleOp> module = ModuleOp::create(kernelLocation);

  Type elementType = kernel.ast.resultType == SourceType::Q15
                         ? static_cast<Type>(builder.getI16Type())
                         : static_cast<Type>(builder.getF32Type());
  MemRefType bufferType = MemRefType::get({ShapedType::kDynamic}, elementType);
  FunctionType functionType = builder.getFunctionType({bufferType, bufferType}, {elementType});
  auto function = func::FuncOp::create(kernelLocation, kernel.ast.name, functionType);
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  llvm::DenseMap<llvm::StringRef, Value> arguments;
  for (auto [parameter, argument] : llvm::zip(kernel.ast.parameters, entry->getArguments()))
    arguments.insert({parameter.name, argument});

  Location expressionLocation = getLocation(context, sourceName, kernel.ast.result.position);
  Value lhs = arguments.lookup(kernel.ast.result.lhs);
  Value rhs = arguments.lookup(kernel.ast.result.rhs);
  if (kernel.ast.resultType == SourceType::Q15) {
    auto numeric = ondsp::FixedAttr::get(&context, ondsp::Signedness::Signed, elementType, 15);
    auto product = ondsp::ProductAttr::get(&context, ondsp::ProductSelection::Full);
    auto accumulatorType = ondsp::AccType::get(&context, builder.getIntegerType(40), 30,
                                               ondsp::Signedness::Signed, *kernel.updateOverflow);
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
    auto result = builder.create<ir::DotOp>(expressionLocation, elementType, lhs, rhs, numeric,
                                            ondsp::ProductAttr());
    builder.create<func::ReturnOp>(expressionLocation, result.getResult());
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
