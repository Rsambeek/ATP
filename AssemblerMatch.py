import sys
import inspect
from typing import Tuple, List, Union, Any, Match, Callable
from functools import reduce

def makeLiteral(token: Union[Token, ASTBranch, Any], variableList: List[dict]) -> Union[Token, Any]:
    if type(token) == ASTBranch:
        token = makeLiteral(runASTBranch(token), variableList)

    if type(token) == Token and token.tokenType == "identifier":
        return Token(variableList[0][token.name].value, "literal")
    else:
        return token

def makeLiteralDecorator(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        newArgs = list(map(lambda x: makeLiteral(x, args[-1]), args[:-1]))
        if "variableList:List[dict]" in str(inspect.signature(func)):
            args = newArgs + args[-1]
        else:
            args = newArgs
        return func(*args, **kwargs)
    return inner

def assignVariable(token1: Token, token2: Token, variableList: List[dict]) -> None:
    returnValue = "\mov " + token1.name + ", " + token2.name + "\n"
    return token1

# Assignments
# newInt :: Token -> List[dict] -> Token
def newInt(identifierToken: Token, variableList: List[dict]) -> Token:
    returnValue = "section .data\n" + identifierToken.name + " dw 0\n" + identifierToken.name + "len equ $ -" + identifierToken.name + "\n\n"
    return identifierToken

# # newFloat :: Token -> List[dict] -> Token
# def newFloat(identifierToken: Token, variableList: List[dict]) -> Token:
#     variableList[0][identifierToken.name] = Variable(identifierToken.name, float)
#     return identifierToken

# # newChar :: Token -> List[dict] -> Token
# def newChar(identifierToken: Token, variableList: List[dict]) -> Token:
#     variableList[0][identifierToken.name] = Variable(identifierToken.name, chr)
#     return identifierToken

# # newString :: Token -> List[dict] -> Token
# def newString(identifierToken: Token, variableList: List[dict]) -> Token:
#     variableList[0][identifierToken.name] = Variable(identifierToken.name, str)
#     return identifierToken

# Key Operations
# ifStatement :: Token -> Union[ASTBranch, None] -> Token
def ifStatement(token1: Token, codeBlock: Union[ASTBranch, None] = None) -> Token:
    returnValue = "mov ax, [" + token1.name + "]\ncmp ax, 0\nje _after" + token1.name + "\n\n"
    returnValue += runASTBranch(codeBlock)
    returnValue += "\n_after" + token1.name + ":\n\n"
    return returnValue

# whileStatement :: Token -> Union[ASTBranch, None] -> List[dict] -> Token
def whileStatement(token1: Token, codeBlock: ASTBranch, environment: List[dict]) -> List[str]:
    if int(makeLiteral(token1, environment).name) > 0:
        newOutput = runASTBranch(codeBlock)
        if type(newOutput) == Token:
            newOutput = ""
        output = whileStatement(token1, codeBlock, environment)
        if newOutput is not None:
            output.insert(0, newOutput)
        return output

    else:
        return [""]


# Operators
# add :: Token -> Token -> Token
@makeLiteralDecorator
def add(token1: Token, token2: Token) -> Token:
    return "ADD r0 r1\n"

# subtract :: Token -> Token -> Token
@makeLiteralDecorator
def subtract(token1: Token, token2: Token) -> Token:
    return "SUB r0 r1\n"

# multiply :: Token -> Token -> Token
@makeLiteralDecorator
def multiply(token1: Token, token2: Token) -> Token:
    return Token(int(token1.name) * int(token2.name))

# devide :: Token -> Token -> Token
@makeLiteralDecorator
def devide(token1: Token, token2: Token) -> Token:
    return Token(int(token1.name) / int(token2.name))

operations = {"int": Function(lambda x, variableScope: newInt(x, variableScope), 1, 10, ["identifier"]),
              "float": Function(lambda x, variableScope: newFloat(x, variableScope), 1, 10, ["identifier"]),
              "char": Function(lambda x, variableScope: newChar(x, variableScope), 1, 10, ["identifier"]),
              "String": Function(lambda x, variableScope: newString(x, variableScope), 1, 10, ["identifier"]),
              "if": Function(lambda x, y, variableScope: ifStatement(makeLiteral(x, variableScope), y), 2, 20, ["identifier", "noRun"]),
              "while": Function(lambda x, y, variableScope: whileStatement(x, y, variableScope), 2, 20, ["identifier", "noRun"]),
              "=": Function(lambda x, y, variableScope: assignVariable(x, makeLiteral(y, variableScope), variableScope), 2, 80, ["identifier"]),
              "+": Function(lambda x, y, variableScope: add(x, y, variableScope), 2, 39, ["identifier"]),
              "-": Function(lambda x, y, variableScope: subtract(x, y, variableScope), 2, 39, ["identifier"]),
              "*": Function(lambda x, y, variableScope: multiply(x, y, variableScope), 2, 38, ["identifier"]),
              "/": Function(lambda x, y, variableScope: devide(x, y, variableScope), 2, 38, ["identifier"])}