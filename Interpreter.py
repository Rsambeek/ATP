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
        return self.name + " : " + self.tokenType

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
def assignVariable(token1: Token, token2: Token, variableList: List[dict]) -> Variable:
    if token2.tokenType == "identifier":
        newValue = variableList[0][token2.name].value
    else:
        newValue = token2.name

    variableList[0][token1.name].value = variableList[0][token1.name].dataType(newValue)
    return token1


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
def add(token1: Token, token2: Token) -> int:
    return int(token1.name) + int(token2.name)



def tokenizeCode(inputCode):
    identifier = {}
    orderOfOperations = {"int": Function(lambda x, variableScope: newInt(x, variableScope), 1, 1, ["identifier"]),
                         "float": Function(lambda x, variableScope: newFloat(x, variableScope), 1, 1, ["identifier"]),
                         "char": Function(lambda x, variableScope: newChar(x, variableScope), 1, 1, ["identifier"]),
                         "String": Function(lambda x, variableScope: newString(x, variableScope), 1, 1, ["identifier"]),
                         "=": Function(lambda x, y, variableScope: assignVariable(x, y, variableScope), 2, 9, ["identifier"])}
    keyword = ["int", "float", "char", "String"]
    separator = [";"]
    operator = ["="]

    environment = {"identifier": identifier, "keyword": keyword, "separator": separator, "operator": operator}

    bracketStack = []
    tokens = []


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


    def lexer(inputCode, currentToken=Token(), bracketStack=[]) -> list:
        if len(inputCode) <= 0:
            return list()
        else:
            if inputCode[0] == " ":
                if len(currentToken.name) != 0:
                    temp = lexer(inputCode[1:], Token(), bracketStack)
                    temp.insert(0, evaluator(currentToken))
                    return temp

            elif inputCode[0] == ";":
                if len(currentToken.name) != 0:
                    temp = lexer(inputCode[1:], Token(), bracketStack)
                    temp.insert(0, evaluator(Token(";")))
                    temp.insert(0, evaluator(currentToken))
                    return temp

            elif inputCode[0] == "'" or inputCode[0] == '"':
                if (len(bracketStack) > 0 and inputCode[0] == bracketStack[-1]):
                    bracketStack.pop()
                    temp = lexer(inputCode[1:], currentToken, bracketStack)
                    temp.insert(0, evaluator(currentToken))
                    return temp
                else:
                    bracketStack.append(inputCode[0])
                    currentToken.tokenType = "literal"

            elif inputCode[0] == "=":
                if currentToken.name == "=":
                    currentToken.name += "="
                else:
                    if currentToken.name != "":
                        return lexer(inputCode[1:], currentToken, bracketStack)
                    else:
                        temp = lexer(inputCode[1:], Token(), bracketStack)
                        temp.insert(0, evaluator(Token("=")))
                        return temp

            elif inputCode[0].isalpha():
                currentToken.name += inputCode[0]
                if currentToken.tokenType == "literal":
                    currentToken.tokenType = ""

            elif inputCode[0].isdigit():
                if currentToken.name == "" and currentToken.tokenType == "":
                    currentToken.tokenType = "literal"
                currentToken.name += inputCode[0]

            temp = lexer(inputCode[1:], currentToken, bracketStack)
            return temp


    # def deduce(token1: Token, token2: Token, identifier: dict):
    #     token1 = deepcopy(token1)
    #     token2 = deepcopy(token2)
    #     print(token1.name + " | " + token2.name)
    #
    #     if token1.tokenType == "keyword":
    #         keyword[token1.name](token2, identifier)
    #         return token2
    #
    #     elif token2.tokenType == "operator":
    #         if token2.name == "=":
    #             token1.operation = "assigning"
    #             return token1
    #         elif token2.name == "+":
    #             token1.operation = "adding"
    #             return token1
    #
    #     elif token1.tokenType == "identifier":
    #         action = None
    #         print(token1)
    #         if token1.operation == "assigning":
    #             action = assignVariable
    #         token1.operation = ""
    #         return action(token1, token2, identifier)
    #
    #     elif token1.tokenType == "literal":
    #         outputToken = Token()
    #         action = None
    #         if token1.operation == "adding":
    #             action = add
    #         else:
    #             return "unexpected operator", True
    #
    #         token1.operation = ""
    #         outputToken.name = action(token1, token2)
    #         outputToken.tokenType = "literal"
    #         return outputToken
    #
    #     else:
    #         print("test")
    #         return ("Failed recognizing token: " + token1.name + " | " + token2.name), True


    def funtionalGetter(orderOfOperations: dict, token: Token) -> Union[Function, Token]:
        if token.name in orderOfOperations:
            return orderOfOperations[token.name]
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

    def treeConstructor(restFunctions: List[Union[Function, Token]]):
        currentHighest = reduce(compareOrder, restFunctions)
        currentHighestIndex = restFunctions.index(currentHighest)
        print(type(currentHighest))
        if type(currentHighest) == Function:
            if currentHighest.parameters == 2:
                if 0 < currentHighestIndex < len(restFunctions):
                    return restFunctions[currentHighestIndex].code(treeConstructor(restFunctions[:currentHighestIndex]),
                                                                   treeConstructor(restFunctions[currentHighestIndex+1:]),
                                                                   list(map(environment.get, currentHighest.environment)))
                else:
                    print("Unexpected Operation")

            else:
                if currentHighestIndex != 0:
                    print("Unexpected Operation")
                return restFunctions[currentHighestIndex].code(treeConstructor(restFunctions[currentHighestIndex+1:]),
                                                               list(map(environment.get, currentHighest.environment)))
        else:
            return currentHighest



    def parser(tokens: List[Token], orderOfOperations: dict):
        functions = list(map(lambda x: funtionalGetter(orderOfOperations, x), tokens))

        AST = treeConstructor(functions)

        return functions


    def runner(tokens: list, identifier=dict(), index=0) -> [Token, None]:
        if index + 1 >= len(tokens):
            output = "\nEnd of code reached\nNo errors encountered\n\nVariableDump:\n"

            for variable in identifier:
                output += (str(identifier[variable].name) + " : " + str(identifier[variable].value) + "  |  ")

            return output, False
        else:
            output = ""
            if tokens[index].tokenType == "keyword":
                identifier = tokens[index].operation(tokens[index + 1], identifier)
                index += 1

            elif tokens[index].tokenType == "operator":
                if index != 0 and len(tokens) > index + 1:
                    identifier = tokens[index].operation(identifier[tokens[index - 1].name], tokens[index + 1], identifier)
                    index += 1
                else:
                    return "unexpected operator", True

            else:
                return ("Failed recognizing token: " + tokens[index].name), True

            nextOutput, error = runner(tokens, identifier, index + 1)
            return (output + nextOutput), False | error
        # if tokenList[0] == ";":
        #     return []
        # else:
        #     return self.runner(tokenList[1:]).insert(0, tokenList[0])


    tokens = lexer(inputCode)
    newFunctions = parser(tokens, orderOfOperations)
    # tokens = list(map(evaluator, tokens))

    output,error = runner(tokens, identifier)


    print(tokens)
    # print(error)
    print(output)
    return True



# sys.setrecursionlimit(0x100000)
# threading.stack_size(256000000)

if not tokenizeCode(inputCode):
    print("Something went wrong")
