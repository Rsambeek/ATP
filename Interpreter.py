import sys
import inspect
from typing import Tuple, List, Union, Any, Match, Callable
from functools import reduce

inputFile = open("codeFile.bimsux", "r")
inputCode = ""
for line in inputFile:
    for word in line:
        for char in word:
            if char != "\n":
                inputCode += char


class BaseObject:
    def printObject(self, name: str, attribute):
        return str(name) + " : " + str(attribute)

class Token(BaseObject):
    name = ""
    tokenType = ""
    operation = None

    def __init__(self, name: str = "", tokenType: str = ""):
        self.name = name
        self.tokenType = tokenType
        self.operation = None

    def __str__(self):
        return self.printObject(self.name, self.tokenType)

class Function(BaseObject):
    code = None
    parameters: int
    order: int
    environment: List[str]

    def __init__(self, code, parameters: int, order: int, environment: List[str] = []):
        self.code = code
        self.parameters = parameters
        self.order = order
        self.environment = environment

    def __str__(self):
        return self.printObject(self.code, self.parameters)


class ASTBranch(BaseObject):
    function = None
    arguments = ()

    def __init__(self, function, arguments: tuple = ()):
        self.function = function
        self.arguments = arguments

    def __str__(self):
        return self.printObject(self.function, self.arguments)


class Variable(BaseObject):
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

    def __str__(self):
        return self.printObject(self.name, self.dataType)


def tokenizeCode(inputCode):
    # Variable Functions
    # makeLiteral :: Union[Token,ASTBranch, Any] -> List[dict] -> Union[Token, Any]
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

    # assignVariable :: Token -> Token -> List[dict] -> None
    def assignVariable(token1: Token, token2: Token, variableList: List[dict]) -> None:
        variableList[0][token1.name].value = variableList[0][token1.name].dataType(token2.name)
        return token1

    # Assignments
    # newInt :: Token -> List[dict] -> Token
    def newInt(identifierToken: Token, variableList: List[dict]) -> Token:
        variableList[0][identifierToken.name] = Variable(identifierToken.name, int)
        return identifierToken

    # newFloat :: Token -> List[dict] -> Token
    def newFloat(identifierToken: Token, variableList: List[dict]) -> Token:
        variableList[0][identifierToken.name] = Variable(identifierToken.name, float)
        return identifierToken

    # newChar :: Token -> List[dict] -> Token
    def newChar(identifierToken: Token, variableList: List[dict]) -> Token:
        variableList[0][identifierToken.name] = Variable(identifierToken.name, chr)
        return identifierToken

    # newString :: Token -> List[dict] -> Token
    def newString(identifierToken: Token, variableList: List[dict]) -> Token:
        variableList[0][identifierToken.name] = Variable(identifierToken.name, str)
        return identifierToken

    # Key Operations
    # ifStatement :: Token -> Union[ASTBranch, None] -> Token
    @makeLiteralDecorator
    def ifStatement(token1: Token, codeBlock: Union[ASTBranch, None] = None) -> Token:
        if int(token1.name) > 0:
            if codeBlock is not None:
                output = None
                output = runASTBranch(codeBlock)
                if type(output) == Token:
                    return output
            return Token(1)
        else:
            return Token(0)

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
        return Token(int(token1.name) + int(token2.name))

    # subtract :: Token -> Token -> Token
    @makeLiteralDecorator
    def subtract(token1: Token, token2: Token) -> Token:
        return Token(int(token1.name) - int(token2.name), "literal")

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
                  "if": Function(lambda x, y, variableScope: ifStatement(x, y, variableScope), 2, 20, ["identifier", "noRun"]),
                  "while": Function(lambda x, y, variableScope: whileStatement(x, y, variableScope), 2, 20, ["identifier", "noRun"]),
                  "=": Function(lambda x, y, variableScope: assignVariable(x, makeLiteral(y, variableScope), variableScope), 2, 80, ["identifier"]),
                  "+": Function(lambda x, y, variableScope: add(x, y, variableScope), 2, 39, ["identifier"]),
                  "-": Function(lambda x, y, variableScope: subtract(x, y, variableScope), 2, 39, ["identifier"]),
                  "*": Function(lambda x, y, variableScope: multiply(x, y, variableScope), 2, 38, ["identifier"]),
                  "/": Function(lambda x, y, variableScope: devide(x, y, variableScope), 2, 38, ["identifier"])}
    identifier = {}
    keyword = ["int", "float", "char", "String", "if"]
    separator = [";", "(", ")", "[", "]", "{", "}"]
    operator = ["=", "+", "-", "*", "/"]

    environment = {"operations": operations,
                   "identifier": identifier,
                   "keyword": keyword,
                   "separator": separator,
                   "operator": operator}


    # Interpreter Steps
    # evaluator :: Token -> Token
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


    # lexer :: str -> Token -> List[List[Token]]
    def lexer(inputCode: str, currentToken=Token()) -> List[List[Token]]:
        if len(inputCode) <= 0:
            return list(list())
        else:
            if inputCode[0] == " ":
                if len(currentToken.name) != 0:
                    temp = lexer(inputCode[1:], Token())
                    temp[0].insert(0, evaluator(currentToken))
                    return temp

            elif inputCode[0] == ";":
                temp = lexer(inputCode[1:], Token())
                temp.insert(0, list())

                if len(currentToken.name) != 0:
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
                    temp = lexer(inputCode[1:], Token())
                    temp[0].insert(0, evaluator(Token(inputCode[0])))
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

    # functionalGetter :: dict -> Token -> Union[Function, Token]
    def functionalGetter(operations: dict, token: Token) -> Union[Function, Token]:
        if token.name in operations:
            return operations[token.name]
        else:
            return token

    # compareOrder :: Union [Function, Token, ASTBranch] -> Union [Function, Token, ASTBranch] -> Union[Function, Token, ASTBranch]
    def compareOrder(operation1: Union[Function, Token, ASTBranch], operation2: Union[Function, Token, ASTBranch]) -> Union[Function, Token, ASTBranch]:
        if type(operation1) == Token and operation1.tokenType == "separator":
            return operation1
        elif type(operation2) == Token and operation2.tokenType == "separator":
            return operation2

        elif type(operation1) == Function and type(operation2) == Function:
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

    # companionBracketFinder :: List[Union[Function, Token]] -> List[chr] -> List[chr] -> Tuple(List[Union[Function, Token]], List[Union[Function, Token]])
    def companionBracketFinder(restList: List[Union[Function, Token]], separators: List[chr], bracketStack: List[chr]=[]) -> (List[Union[Function, Token]], List[Union[Function, Token]]):
        bracketPairs = {"(": ")", "[": "]", "{": "}"}
        if type(restList) == Token:
            return restList

        if len(bracketStack) == 0:
            bracketStack.append(bracketPairs[restList[0].name])
            return companionBracketFinder(restList[1:], separators, bracketStack)

        else:
            if type(restList[0]) == Token:
                currentItem = restList[0].name
                if currentItem in separators:
                    if currentItem in bracketPairs:
                        bracketStack.append(bracketPairs[currentItem])

                    elif bracketStack[-1] == currentItem:
                        bracketStack.pop()
                if len(bracketStack) <= 0:
                    return [], restList[1:]

            temp = companionBracketFinder(restList[1:], separators, bracketStack)
            return [restList[0]] + temp[0], temp[1]

    # treeConstructor :: List[Union[Function, Token, ASTBranch]] -> dict -> Union[ASTBranch, Token]
    def treeConstructor(restFunctions: List[Union[Function, Token, ASTBranch]], environment: dict) -> Union[ASTBranch, Token]:
        currentHighest = reduce(compareOrder, restFunctions)
        currentHighestIndex = restFunctions.index(currentHighest)

        if type(currentHighest) == ASTBranch:
            return currentHighest

        elif type(currentHighest) == Function:
            if currentHighest.parameters == 2:
                if 0 < currentHighestIndex < len(restFunctions):
                    return ASTBranch(restFunctions[currentHighestIndex].code,
                                     (treeConstructor(restFunctions[:currentHighestIndex], environment),
                                      treeConstructor(restFunctions[currentHighestIndex+1:], environment),
                                      list(map(environment.get, currentHighest.environment))))
                else:
                    return ASTBranch(restFunctions[currentHighestIndex].code,
                                     (restFunctions[currentHighestIndex + 1],
                                      restFunctions[currentHighestIndex + 2],
                                      list(map(lambda x: environment.get(x) if (x in environment) else x, currentHighest.environment))))

            else:
                if currentHighestIndex != 0:
                    print("Unexpected Operation")
                return ASTBranch(restFunctions[currentHighestIndex].code,
                                 (treeConstructor(restFunctions[currentHighestIndex+1:], environment),
                                  list(map(environment.get, currentHighest.environment))))

        elif currentHighest.tokenType == "separator":
            temp = companionBracketFinder(restFunctions[currentHighestIndex:],environment["separator"])
            restFunctions = restFunctions[:currentHighestIndex] + [treeConstructor(temp[0], environment)] + temp[1]
            return treeConstructor(restFunctions, environment)

        else:
            return currentHighest

    # parser :: List[List[Token]] -> dict -> List[Union[ASTBranch, Token]]
    def parser(tokens: List[List[Token]], environment: dict) -> List[Union[ASTBranch, Token]]:
        functions = list(map(lambda innerList: list(map(lambda item: functionalGetter(environment["operations"], item), innerList)), tokens))
        AST = list(map(lambda currentExpression: treeConstructor(currentExpression, environment), functions))

        return AST

    # runASTBranch :: Union[ASTBranch, Token] -> Union[Token, str]
    def runASTBranch(input: Union[ASTBranch, Token]) -> Union[Token, str]:
        if type(input) != ASTBranch:
            return input
        else:
            if "noRun" in input.arguments[-1]:
                input.arguments[-1].pop(input.arguments[-1].index("noRun"))
                return input.function(*input.arguments)

            else:
                return input.function(*list(map(runASTBranch, input.arguments)))

    # runner:: List[ASTBranch] -> ([str], bool)
    def runner(functions: List[ASTBranch]) -> ([str], bool):
        if len(functions) <= 0:
            output = "\nEnd of code reached\nNo errors encountered\n\nVariableDump:\n"
            return [output], False
        else:
            newOutput = runASTBranch(functions[0])
            if type(newOutput) == Token:
                newOutput = ""
            output = runner(functions[1:])
            if type(newOutput) == list:
                return newOutput + output[0], output[1]
            else:
                output[0].insert(0, newOutput)
                return output


    tokens = lexer(inputCode)
    # print(tokens)
    AST = parser(tokens, environment)
    output = runner(AST)
    output = [reduce(lambda x,y: x+y,output[0]), output[1]]


    # print(tokens)
    for variable in environment["identifier"]:
        output[0] += (str(identifier[variable].dataType) + " " + str(identifier[variable].name) + " : " + str(identifier[variable].value) + "  |  ")
    return output



sys.setrecursionlimit(0x100000)
# threading.stack_size(256000000)

output = tokenizeCode(inputCode)
if output[1]:
    print("Something went wrong")

print(output[0])