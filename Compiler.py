import sys
import inspect
from typing import Tuple, List, Union, Any, Match, Callable
from functools import reduce

from architecture import architecture, x86 as architect
# from architecture import cortex as architect

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
    # functionHeader :: str -> str -> str
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

    # registerPrepDecorator :: Callable -> Callable
    def registerPrepDecorator(func: Callable) -> Callable:
        def inner(*args, **kwargs):
            index = 0
            returnValue = ""
            for arg in args:
                if type(arg) == Token:
                    returnValue += "mov " + architect.REGISTERS[index] + ", " + arg.name + "\n"
                    index += 1
                elif type(arg) == ASTBranch:
                    returnValue += compileASTBranch(arg)
            
            returnValue += func(*args, **kwargs)
            return returnValue
        return inner
    
    # prepareAsmStatement :: Union[str, None] -> str
    def prepareAsmStatement(input: Union[str, None]) -> str:
        if type(input) == str:
            index = input.find("section .text\n")
            if index == -1:
                output = input + "section .text\n"
        else:
            output = "section .text\n"
        return output

    # assignVariable :: Token -> Token -> List[dict] -> Token
    def assignVariable(token1: Token, token2: Token) -> Token:
        # print(token1)
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)

        returnToken.operation += "mov " + architect.R0 + ", " + token2.name + "\n"
        returnToken.operation += "mov [" + token1.name + "], " + architect.R0 + "\n"
        return returnToken

    # Definitions
    # newInt :: Token -> List[dict] -> Token
    def newInt(identifierToken: Token) -> Token:
        returnToken = Token(identifierToken.name)
        newAssembly = identifierToken.name + " dw 0\n" + identifierToken.name + "len equ $ - " + identifierToken.name
        if type(identifierToken.operation) == str:
            index = identifierToken.operation.find("section .data\n")
            if index != -1:
                returnToken.operation = identifierToken.operation[:index + 14] + newAssembly + identifierToken.operation[index + 14:]
            else:
                returnToken.operation = "section .data\n" + newAssembly + identifierToken.operation
        else:
            returnToken.operation = "section .data\n" + newAssembly

        returnToken.operation += "\n"
        # print("::" + returnToken.operation)
        return returnToken

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
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "mov " + architect.R0 + ", [" + token1.name + "]\ncmp " + architect.R0 + ", 0\nje _afterif" + token1.name + "\n\n"
        returnToken.operation = asmMerger(returnToken.operation, compileASTBranch(codeBlock).operation)
        returnToken.operation += "\n_afterif" + token1.name + ":\n\n"
        return returnToken

    # whileStatement :: Token -> Union[ASTBranch, None] -> List[dict] -> Token
    def whileStatement(token1: Token, codeBlock: ASTBranch) -> List[str]:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "mov " + architect.R0 + ", [" + token1.name + "]\ncmp " + architect.R0 + ", 0\nje _afterwhile" + token1.name + "\n_startwhile" + token1.name + "\n\n"
        returnToken.operation = asmMerger(returnToken.operation, compileASTBranch(codeBlock).operation)
        returnToken.operation += "mov " + architect.R0 + ", [" + token1.name + "]\ncmp " + architect.R0 + ", 0\njne _startwhile" + token1.name + "\n"
        returnToken.operation += "\n_afterwhile" + token1.name + ":\n\n"
        return returnToken


    # Operators
    # add :: Token -> Token -> Token
    @registerPrepDecorator
    def add(token1: Token, token2: Token) -> Token:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "mov " + architect.R0 + ", " + token1.name + "\n"
        returnToken.operation += "mov " + architect.R1 + ", " + token2.name + "\n"
        returnToken.operation += "add " + architect.R0 + ", " + architect.R1 + "\n"
        return returnToken

    # subtract :: Token -> Token -> Token
    @registerPrepDecorator
    def subtract(token1: Token, token2: Token) -> Token:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "mov " + architect.R0 + ", " + token1.name + "\n"
        returnToken.operation += "mov " + architect.R1 + ", " + token2.name + "\n"
        returnToken.operation += "sub " + architect.R0 + ", " + architect.R1 + "\n"
        return returnToken

    # # multiply :: Token -> Token -> Token
    # @registerPrepDecorator
    # def multiply(token1: Token, token2: Token) -> Token:
    #     returnValue = "mul " + R1 + "\n"
    #     return returnValue

    # # devide :: Token -> Token -> Token
    # @registerPrepDecorator
    # def devide(token1: Token, token2: Token) -> Token:
    #     returnValue = "div " + R1 + "\n"
    #     return returnValue

    operations = {"int": Function(lambda x, variableScope: newInt(x), 1, 10, ["identifier"]),
                # "float": Function(lambda x, variableScope: newFloat(x), 1, 10, ["identifier"]),
                # "char": Function(lambda x, variableScope: newChar(x), 1, 10, ["identifier"]),
                # "String": Function(lambda x, variableScope: newString(x), 1, 10, ["identifier"]),
                "if": Function(lambda x, y, variableScope: ifStatement(x, y), 2, 20, ["identifier"]),
                "while": Function(lambda x, y, variableScope: whileStatement(x, y), 2, 20, ["identifier"]),
                "=": Function(lambda x, y, variableScope: assignVariable(x, y), 2, 80, ["identifier"]),
                "+": Function(lambda x, y, variableScope: add(x, y), 2, 39, ["identifier"]),
                "-": Function(lambda x, y, variableScope: subtract(x, y), 2, 39, ["identifier"])}
                # "*": Function(lambda x, y, variableScope: multiply(x, y), 2, 38, ["identifier"]),
                # "/": Function(lambda x, y, variableScope: devide(x, y), 2, 38, ["identifier"])}
    identifier = {}
    keyword = ["int", "float", "char", "String", "if", "while"]
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

    # compileASTBranch :: Union[ASTBranch, Token] -> str
    def compileASTBranch(input: Union[ASTBranch, Token]) -> str:
        if type(input) == Token:
            return input

        elif type(input) == ASTBranch:
            if "noRun" in input.arguments[-1]:
                input.arguments[-1].pop(input.arguments[-1].index("noRun"))
                return input.function(*input.arguments)

            else:
                return input.function(*list(map(compileASTBranch, input.arguments)))

        else:
            return input
    
    # asmMerger :: str -> str -> str
    def asmMerger(code: str, newCode: str) -> str:
        codeTextIndex = code.find("section .text")
        codeDataIndex = code.find("section .data")
        newCodeTextIndex = newCode.find("section .text")
        newCodeDataIndex = newCode.find("section .data")
        returnValue = ""
        # print("::----")
        # print(code)
        # print("******")
        # print(newCode)
        # print("------")

        if codeDataIndex != -1 or newCodeDataIndex != -1:
            returnValue += "section .data\n"
    
        if codeDataIndex != -1:
            returnValue += code[codeDataIndex+14:codeTextIndex]
            # print("1 " + code[codeDataIndex+14:codeTextIndex])
            # print("----")
        if newCodeDataIndex != -1:
            returnValue += newCode[newCodeDataIndex+14:newCodeTextIndex]
            # print("2 " + newCode[newCodeDataIndex+14:newCodeTextIndex])
            # print("----")
        
        if len(returnValue) > 0 and returnValue[-2:] != "\n\n":
            if returnValue[-1] != "\n":
                returnValue += "\n"
            returnValue += "\n"
        
        if codeTextIndex != -1 or newCodeTextIndex != -1:
            returnValue += "section .text\n"
        if codeTextIndex != -1:
            returnValue += code[codeTextIndex+14:]
            # print("3 " + code[codeTextIndex+14:])
            # print("----")
        if newCodeTextIndex != -1:
            returnValue += newCode[newCodeTextIndex+14:]
            # print("4 " + newCode[newCodeTextIndex+14:])
        return returnValue

    # compile:: List[ASTBranch] -> Tuple[str, bool]
    def compile(functions: List[ASTBranch]) -> Tuple[str, bool]:
        if len(functions) <= 0:
            output = ""
            # output = "\nEnd of code reached\nNo errors encountered\n\nVariableDump:\n"
            return output, False
        else:
            newOutput = compileASTBranch(functions[0])
            if type(newOutput) == Token:
                if newOutput.operation != None:
                    newOutput = newOutput.operation
                else:
                    newOutput = newOutput.name
            # print(newOutput)
            output = compile(functions[1:])
            if type(newOutput) == list:
                return newOutput + output[0], output[1]
            else:
                newOutput = asmMerger(newOutput, output[0])
                # print(newOutput)
                # print("----")
                # output[0].insert(0, newOutput)
                return newOutput, output[1]


    tokens = lexer(inputCode)
    # print(tokens)
    AST = parser(tokens, environment)
    output = compile(AST)
    startText = "section .text\nglobal _start\n_start:\n"
    endText = "section .text\nmov eax, 1\nint 0x80"
    outputText = asmMerger(startText, output[0])
    outputText = asmMerger(outputText, endText)
    output = outputText, output[1]


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