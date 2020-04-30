import sys
import threading
from copy import deepcopy
from typing import Tuple, List, Union, Any, Match
from functools import reduce

inputFile = open("codeFile.bimsux", "r")
inputCode = ""
for line in inputFile:
    for word in line:
        for char in word:
            if char != "\n":
                inputCode += char


class Token:
    name = ""
    tokenType = ""
    operation = None

    def __init__(self, name: str = "", tokenType: str = ""):
        self.name = name
        self.tokenType = tokenType
        self.operation = None

    def __repr__(self):
        return "<class 'Token_" + self.name + ":" + self.tokenType + "'>"

    def __str__(self):
        return str(self.name) + " : " + str(self.tokenType)

class Function:
    code = None
    parameters: int
    order: int
    environment: List[str]

    def __init__(self, code, parameters: int, order: int, environment: List[str] = []):
        self.code = code
        self.parameters = parameters
        self.order = order
        self.environment = environment


class ASTBranch:
    function = None
    arguments = ()

    def __init__(self, function, arguments: tuple = ()):
        self.function = function
        self.arguments = arguments


class Variable:
    name: str
    dataType: type
    value: type

    def __init__(self, name: str, dataType: type, value: type = None):
        self.name = name
        self.dataType = dataType
        if value is not None:
            self.value = value
        else:
            self.value = self.dataType()

    def __repr__(self):
        return "<" + str(self.dataType) + ":" + self.name + ">"

    def __str__(self):
        return self.name + " : " + str(self.dataType)


# Variable Functions
def makeLiteral(token: Token, variableList: List[dict]) -> Token:
    if token.tokenType == "identifier":
        return Token(variableList[0][token.name].value, "literal")
    else:
        return token


def assignVariable(token1: Token, token2: Token, variableList: List[dict]) -> None:
    variableList[0][token1.name].value = variableList[0][token1.name].dataType(token2.name)
    return


# Assignments
def newInt(identifierToken: Token, variableList: List[dict]) -> Token:
    variableList[0][identifierToken.name] = Variable(identifierToken.name, int)
    return identifierToken
def newFloat(identifierToken: Token, variableList: List[dict]) -> Token:
    variableList[0][identifierToken.name] = Variable(identifierToken.name, float)
    return identifierToken
def newChar(identifierToken: Token, variableList: List[dict]) -> Token:
    variableList[0][identifierToken.name] = Variable(identifierToken.name, chr)
    return identifierToken
def newString(identifierToken: Token, variableList: List[dict]) -> Token:
    variableList[0][identifierToken.name] = Variable(identifierToken.name, str)
    return identifierToken

# Operators
def add(token1: Token, token2: Token) -> Token:
    return Token(int(token1.name) + int(token2.name))
def subtract(token1: Token, token2: Token) -> Token:
    return Token(int(token1.name) - int(token2.name))
def multiply(token1: Token, token2: Token) -> Token:
    return Token(int(token1.name) * int(token2.name))
def devide(token1: Token, token2: Token) -> Token:
    return Token(int(token1.name) / int(token2.name))



def tokenizeCode(inputCode):
    operations = {"int": Function(lambda x, variableScope: newInt(x, variableScope), 1, 10, ["identifier"]),
                  "float": Function(lambda x, variableScope: newFloat(x, variableScope), 1, 10, ["identifier"]),
                  "char": Function(lambda x, variableScope: newChar(x, variableScope), 1, 10, ["identifier"]),
                  "String": Function(lambda x, variableScope: newString(x, variableScope), 1, 10, ["identifier"]),
                  "=": Function(lambda x, y, variableScope: assignVariable(x, makeLiteral(y, variableScope), variableScope), 2, 90, ["identifier"]),
                  "+": Function(lambda x, y, variableScope: add(makeLiteral(x, variableScope), makeLiteral(y, variableScope)), 2, 29, ["identifier"]),
                  "-": Function(lambda x, y, variableScope: subtract(makeLiteral(x, variableScope), makeLiteral(y, variableScope)), 2, 29, ["identifier"]),
                  "*": Function(lambda x, y, variableScope: multiply(makeLiteral(x, variableScope), makeLiteral(y, variableScope)), 2, 28, ["identifier"]),
                  "/": Function(lambda x, y, variableScope: devide(makeLiteral(x, variableScope), makeLiteral(y, variableScope)), 2, 28, ["identifier"]),
                  "(": Function(lambda x, y, variableScope: devide(makeLiteral(x, variableScope), makeLiteral(y, variableScope)), 2, 28, ["identifier"]),
                  "[": Function(lambda x, y, variableScope: devide(makeLiteral(x, variableScope), makeLiteral(y, variableScope)), 2, 28, ["identifier"]),
                  "{": Function(lambda x, y, variableScope: devide(makeLiteral(x, variableScope), makeLiteral(y, variableScope)), 2, 28, ["identifier"])}
    identifier = {}
    keyword = ["int", "float", "char", "String"]
    separator = {";": ";", "(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}
    operator = ["=", "+", "-", "*", "/"]

    environment = {"operations": operations,
                   "identifier": identifier,
                   "keyword": keyword,
                   "separator": separator,
                   "operator": operator}


    # Interpreter Steps
    def evaluator(token: Token) -> Token:
        if token.name != "":
            if token.tokenType == "":
                if token.name in keyword:
                    token.tokenType = "keyword"
                elif token.name in separator:
                    token.tokenType = "separator"
                elif token.name in operator:
                    token.tokenType = "operator"

                else:
                    token.tokenType = "identifier"

            return token


    def lexer(inputCode, currentToken=Token()) -> List[List[Token]]:
        if len(inputCode) <= 0:
            return list(list())
        else:
            if inputCode[0] == " ":
                if len(currentToken.name) != 0:
                    temp = lexer(inputCode[1:], Token())
                    temp[0].insert(0, evaluator(currentToken))
                    return temp

            elif inputCode[0] == ";":
                if len(currentToken.name) != 0:
                    temp = lexer(inputCode[1:], Token())
                    temp.insert(0, list())
                    temp[0].insert(0, evaluator(currentToken))
                    return temp

            # elif inputCode[0] == "'" or inputCode[0] == '"':
            #     bracketStack.append(inputCode[0])
            #
            #     temp = lexer(inputCode[1:], Token(), bracketStack)
            #     temp[0].insert(0, evaluator(currentToken))
            #
            #     currentToken.tokenType = "literal"

            elif inputCode[0] in operator or inputCode[0] in separator:
                if currentToken.name != "":
                    temp = lexer(inputCode[0:], Token())
                    temp[0].insert(0, evaluator(currentToken))
                else:
                    temp = lexer(inputCode[1:], Token())
                    temp[0].insert(0, evaluator(Token(inputCode[0])))

                return temp

            elif inputCode[0].isalpha():
                currentToken.name += inputCode[0]
                if currentToken.tokenType == "literal":
                    currentToken.tokenType = ""

            elif inputCode[0].isdigit():
                if currentToken.name == "" and currentToken.tokenType == "":
                    currentToken.tokenType = "literal"
                currentToken.name += inputCode[0]

        temp = lexer(inputCode[1:], currentToken)
        return temp

    def functionalGetter(operations: dict, token: Token) -> Union[Function, Token]:
        if token.name in operations:
            return operations[token.name]
        else:
            return token

    def compareOrder(operation1: Union[Function, Token], operation2: Union[Function, Token]):
        if type(operation1) == Function and type(operation2) == Function:
            if operation1.order >= operation2.order:
                return operation1
            else:
                return operation2

        elif type(operation1) == Function:
            return operation1

        elif type(operation2) == Function:
            return operation2

        else:
            return operation1

    def bracketFinder(restList: List[Union[Function, Token]], environment: dict, bracketStack: List[chr]=[]) -> List[Union[Function, Token]]:
        print(bracketStack, restList)
        if len(bracketStack) == 0:
            bracketStack.append(restList[0])
            return bracketFinder(restList[1:], environment, bracketStack)

        else:
            if restList[0] in environment["separator"]:
                bracketStack.append(restList[0])

            elif bracketStack[-1] == restList[0]:
                bracketStack.pop()
                if len(bracketStack) <= 0:
                    return []

            return [restList[0]] + bracketFinder(restList[1:], environment, bracketStack)

    def treeConstructor(restFunctions: List[Union[Function, Token]], environment: dict):
        currentHighest = reduce(compareOrder, restFunctions)
        currentHighestIndex = restFunctions.index(currentHighest)
        if type(currentHighest) == Function:
            if currentHighest.parameters == 2:
                if 0 < currentHighestIndex < len(restFunctions):
                    return ASTBranch(restFunctions[currentHighestIndex].code,
                                     (treeConstructor(restFunctions[:currentHighestIndex], environment),
                                      treeConstructor(restFunctions[currentHighestIndex+1:], environment),
                                      list(map(environment.get, currentHighest.environment))))
                else:
                    print("Unexpected Operation")

            else:
                if currentHighestIndex != 0:
                    print("Unexpected Operation")
                return ASTBranch(restFunctions[currentHighestIndex].code,
                                 (treeConstructor(restFunctions[currentHighestIndex+1:], environment),
                                  list(map(environment.get, currentHighest.environment))))
        elif currentHighest.tokenType == "separator":
            return ASTBranch(treeConstructor(bracketFinder(restFunctions,environment), environment))

        else:
            return currentHighest



    def parser(tokens: List[List[Token]], environment: dict):
        functions = list(map(lambda innerList: list(map(lambda item: functionalGetter(environment["operations"], item), innerList)), tokens))
        AST = list(map(lambda currentExpression: treeConstructor(currentExpression, environment), functions))

        return AST

    def runASTBranch(input: Union[ASTBranch, Token]):
        # print(input)
        if type(input) != ASTBranch:
            return input
        else:
            if input.arguments[-1] == []:
                return input.function(*list(map(runASTBranch, input.arguments[:-1])))
            else:
                return input.function(*list(map(runASTBranch, input.arguments)))

    def runner(functions: List[ASTBranch]) -> [str]:
        if len(functions) <= 0:
            output = "\nEnd of code reached\nNo errors encountered\n\nVariableDump:\n"
            return [output], False
        else:
            newOutput = runASTBranch(functions[0])
            output = runner(functions[1:])
            if newOutput is not None:
                output[0].insert(0, newOutput)
            return output


    tokens = lexer(inputCode)
    print(tokens)
    AST = parser(tokens, environment)
    output = runner(AST)
    output = [reduce(lambda x,y: x+y,output[0]), output[1]]


    # print(tokens)
    for variable in environment["identifier"]:
        output[0] += (str(identifier[variable].dataType) + " " + str(identifier[variable].name) + " : " + str(identifier[variable].value) + "  |  ")
    return output



# sys.setrecursionlimit(0x100000)
# threading.stack_size(256000000)

output = tokenizeCode(inputCode)
if output[1]:
    print("Something went wrong")

print(output[0])