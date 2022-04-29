import sys
import inspect
from typing import Tuple, List, Union, Any, Match, Callable
from functools import reduce

from architecture import x86 as architect
# from architecture import cortex as architect

def functionHeader(name: str, innerFunction: str) -> str:
    returnValue = name + ":\npush {"
    for register in architect.REGISTERS[architect.SAFEINDEX:]:
        returnValue += register + ", "
    
    returnValue += "lr}\n"
    returnValue += innerFunction
    returnValue += "pop {"
    for register in architect.REGISTERS[architect.SAFEINDEX:]:
        returnValue += register + ", "
    
    returnValue += "pc}\n"
    return "push"

def registerPrepDecorator(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        index = 0
        returnValue = ""
        for arg in args:
            returnValue += "mov " + architect.REGISTERS[index] + ", " + arg.name + "\n"
            index += 1
        
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
    returnValue = "mov " + architect.R0 + ", [" + token1.name + "]\ncmp " + architect.R0 + ", 0\nje _afterif" + token1.name + "\n\n"
    returnValue += runASTBranch(codeBlock)
    returnValue += "\n_afterif" + token1.name + ":\n\n"
    return returnValue

# whileStatement :: Token -> Union[ASTBranch, None] -> List[dict] -> Token
def whileStatement(token1: Token, codeBlock: ASTBranch, environment: List[dict]) -> List[str]:
    returnValue = "mov " + architect.R0 + ", [" + token1.name + "]\ncmp " + architect.R0 + ", 0\nje _afterwhile" + token1.name + "\n_startwhile" + token1.name + "\n\n"
    returnValue += runASTBranch(codeBlock)
    returnValue = "mov " + architect.R0 + ", [" + token1.name + "]\ncmp " + architect.R0 + ", 0\njne _startwhile" + token1.name + "\n"
    returnValue += "\n_afterwhile" + token1.name + ":\n\n"


# Operators
# add :: Token -> Token -> Token
@registerPrepDecorator
def add(token1: Token, token2: Token) -> Token:
    returnValue = "add " + architect.R0 + ", " + architect.R1 + "\n"
    return returnValue

# subtract :: Token -> Token -> Token
@registerPrepDecorator
def subtract(token1: Token, token2: Token) -> Token:
    returnValue = "sub " + architect.R0 + ", " + architect.R1 + "\n"
    return returnValue

# # multiply :: Token -> Token -> Token
# @makeLiteralDecorator
# def multiply(token1: Token, token2: Token) -> Token:
#     returnValue = "mul " + R1 + "\n"
#     return returnValue

# # devide :: Token -> Token -> Token
# @makeLiteralDecorator
# def devide(token1: Token, token2: Token) -> Token:
#     returnValue = "div " + R1 + "\n"
#     return returnValue

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