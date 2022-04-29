from platform import architecture
import sys
import inspect
from typing import Tuple, List, Union, Any, Match, Callable
from functools import reduce

# from architecture import architecture, x86 as architect
from architecture import cortex as architect

# Read data from file
inputFile = open("codeFile.bimsux", "r")
inputCode = ""
for line in inputFile:
    for word in line:
        for char in word:
            if char != "\n":
                inputCode += char

# Base object for printing objects
class BaseObject:
    def printObject(self, name: str, attribute):
        return str(name) + " : " + str(attribute)

class Token(BaseObject):
    name = ""
    tokenType = ""
    operation = None

    # Initializes token and fills data
    def __init__(self, name: str = "", tokenType: str = ""):
        self.name = name
        self.tokenType = tokenType
        self.operation = None

    def __str__(self):
        return self.printObject(self.name, self.tokenType)

# Class for function deduction
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

# Class for building AST Tree
class ASTBranch(BaseObject):
    function = None
    arguments = ()

    def __init__(self, function, arguments: tuple = ()):
        self.function = function
        self.arguments = arguments

    def __str__(self):
        return self.printObject(self.function, self.arguments)

# Class for datastructure to store data
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
    # registerPrepDecorator :: Callable -> Callable
    def registerPrepDecorator(func: Callable) -> Callable:
        # Function for prepering registers for function
        def inner(*args, **kwargs):
            index = 0
            returnValue = prepareAsmStatement("")
            for arg in args:
                if type(arg) == Token:
                    if (arg.tokenType == "literal"):
                        returnValue += "\tmov " + architect.REGISTERS[index] + ", #" + arg.name + "\n"
                    elif (arg.tokenType == "identifier"):
                        returnValue += "\tldr " + architect.R4 + ", =" + arg.name + "\n"
                        returnValue += "\tldr " + architect.REGISTERS[index] + ", [" + architect.R4 + "]\n"
                elif type(arg) == ASTBranch:
                    returnValue = asmMerger(returnValue, compileASTBranch(arg))
                index += 1
            returnedValue = func(*args, **kwargs)
            
            if type(returnedValue) == Token:
                returnValue = asmMerger(returnValue, returnedValue.operation)
            elif type(returnedValue) == str:
                returnValue = asmMerger(returnValue, returnedValue)
            returnToken = Token(returnValue)
            returnToken.operation = returnValue
            return returnToken
        return inner
    
    # prepareAsmStatement :: Union[str, None] -> str
    def prepareAsmStatement(input: Union[str, None]) -> str:
        # Insert needed texts for compilation
        if type(input) == str:
            index = input.find(".text\n")
            if index == -1:
                output = input + ".text\n"
        else:
            output = ".text\n"
        return output

    # assignVariable :: Token -> Token -> Token
    def assignVariable(token1: Token, token2: Token) -> Token:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)

        if (token2.tokenType == "literal"):
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token2.name + "\n"
        elif (token2.tokenType == "identifier"):
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token2.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        else:
            returnToken.operation = asmMerger(returnToken.operation, token2.name)
        returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
        returnToken.operation += "\tstr " + architect.R0 + ", [" + architect.R1 + "]" + "\n"
        return returnToken

    # Definitions
    # newInt :: Token -> List[dict] -> Token
    def newInt(identifierToken: Token) -> Token:
        returnToken = Token(identifierToken.name)
        newAssembly = "\t" + identifierToken.name + ": .space 4\n"
        # newAssembly = "\t" + identifierToken.name + " dw 0\n\t" + identifierToken.name + "len equ $ - " + identifierToken.name
        if type(identifierToken.operation) == str:
            index = identifierToken.operation.find(".data\n")
            if index != -1:
                returnToken.operation = identifierToken.operation[:index + 6] + newAssembly + identifierToken.operation[index + 6:]
            else:
                returnToken.operation = ".data\n" + newAssembly + identifierToken.operation
        else:
            returnToken.operation = ".data\n" + newAssembly

        # returnToken.operation += "\n"
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

    # functionHeader :: Token -> Union[List, Token, None] -> Union[ASTBranch, None] -> Token
    def functionHeader(token1: Token, parameters: Union[List, Token, None] = None, codeBlock: Union[ASTBranch, None] = None) -> Token:
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)

        # print(parameters)

        returnToken.operation += "\n\t.global " + token1.name + "\n"
        returnToken.operation += "\tb _after" + token1.name + "\n"
        returnToken.operation += token1.name + ":\n"

        if (type(parameters) == Token):
            parameters = [parameters]
        
        for i in range(len(parameters)):
            # parameters[i].operation = returnToken.operation
            # print(newInt(parameters[i]).operation)
            newAsm = newInt(parameters[i]).operation
            if (returnToken.operation.find(newAsm) == -1):
                returnToken.operation = asmMerger(returnToken.operation, newAsm)
            
            returnToken.operation += "\tldr " + architect.R4 + ", =" + parameters[i].name + "\n"
            returnToken.operation += "\tldr " + architect.REGISTERS[i] + ", [" + architect.R4 + "]\n"

        returnToken.operation = asmMerger(returnToken.operation, compileASTBranch(codeBlock).operation)
        returnToken.operation += "_after" + token1.name + ":\n"
        
        # pop stacked back to registers
        returnToken.operation += "\tpop {"
        for register in architect.REGISTERS[architect.SAFEINDEX:]:
            returnToken.operation += register + ", "
        
        returnToken.operation += "PC}\n\n"
        return returnToken
    
    # callFunction :: Token -> Token
    def callFunction(token1: Token) -> Token:
        # Fucntion for calling predefined function blocks
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)

        # push registers to stack
        returnToken.operation += "\tpush {"
        for register in architect.REGISTERS[architect.SAFEINDEX:]:
            returnToken.operation += register + ", "
        
        returnToken.operation += "lr}\n"
        returnToken.operation += "\tb " + token1.name + "\n"
        return returnToken
    
    # returnFunction :: Token -> Token
    def returnFunction(token1 : Token) -> Token:
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        # Return argument from function by storing it in register 0
        returnToken.operation = prepareAsmStatement(returnToken.operation)

        if (token1.tokenType == "literal"):
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        return returnToken

    # Key Operations
    # ifStatement :: Token -> Union[ASTBranch, None] -> Token
    def ifStatement(token1: Token, codeBlock: Union[ASTBranch, None] = None) -> Token:
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "\n"

        # assess execute condition
        returnedValue = compileASTBranch(codeBlock).operation
        functionHash = hash(returnedValue)
        functionHash += sys.maxsize + 1
        functionHash = token1.name + str(functionHash)

        if (token1.tokenType == "literal"):
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        
        returnToken.operation += "\tcmp " + architect.R0 + ", #0\n\tbeq _afterif" + functionHash + "\n"
        returnToken.operation = asmMerger(returnToken.operation, returnedValue)
        returnToken.operation += "_afterif" + functionHash + ":\n\n"
        return returnToken

    # Key Operations
    # ifnStatement :: Token -> Union[ASTBranch, None] -> Token
    def ifnStatement(token1: Token, codeBlock: Union[ASTBranch, None] = None) -> Token:
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "\n"

        # assess execute condition
        returnedValue = compileASTBranch(codeBlock).operation
        functionHash = hash(returnedValue)
        functionHash += sys.maxsize + 1
        functionHash = token1.name + str(functionHash)

        if (token1.tokenType == "literal"):
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        
        returnToken.operation += "\tcmp " + architect.R0 + ", #0\n\tbne _afterif" + functionHash + "\n"
        returnToken.operation = asmMerger(returnToken.operation, returnedValue)
        returnToken.operation += "_afterif" + functionHash + ":\n\n"
        return returnToken

    # whileStatement :: Token -> ASTBranch -> List[dict] -> Token
    def whileStatement(token1: Token, codeBlock: ASTBranch) -> List[str]:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "\n"

        # assess run condition
        if (token1.tokenType == "literal"):
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        
        returnToken.operation += "\tcmp " + architect.R0 + ", #0\tbeq _afterwhile" + token1.name + "\n_startwhile" + token1.name + ":\n\n"
        returnToken.operation = asmMerger(returnToken.operation, compileASTBranch(codeBlock).operation)

        if (token1.tokenType == "literal"):
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        returnToken.operation += "\tcmp " + architect.R0 + ", #0\n\tbne _startwhile" + token1.name + "\n"
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
        
        # returnToken.operation += "mov " + architect.R0 + ", " + token1.name + "\n"
        # returnToken.operation += "mov " + architect.R1 + ", " + token2.name + "\n"
        returnToken.operation += "\tadd " + architect.R0 + ", " + architect.R1 + "\n"
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
        
        # returnToken.operation += "mov " + architect.R0 + ", " + token1.name + "\n"
        # returnToken.operation += "mov " + architect.R1 + ", " + token2.name + "\n"
        returnToken.operation += "\tsub " + architect.R0 + ", " + architect.R1 + "\n"
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
                "func": Function(lambda x, y, z, variableScope: functionHeader(x, y, z), 3, 10, ["identifier"]),
                "call": Function(lambda x, variableScope: callFunction(x), 1, 10, ["identifier"]),
                "return": Function(lambda x, variableScope: returnFunction(x), 1, 10, ["identifier"]),
                "if": Function(lambda x, y, variableScope: ifStatement(x, y), 2, 20, ["identifier"]),
                "ifn": Function(lambda x, y, variableScope: ifnStatement(x, y), 2, 20, ["identifier"]),
                "while": Function(lambda x, y, variableScope: whileStatement(x, y), 2, 20, ["identifier"]),
                "=": Function(lambda x, y, variableScope: assignVariable(x, y), 2, 80, ["identifier"]),
                "+": Function(lambda x, y, variableScope: add(x, y), 2, 39, ["identifier"]),
                "-": Function(lambda x, y, variableScope: subtract(x, y), 2, 39, ["identifier"])}
                # "*": Function(lambda x, y, variableScope: multiply(x, y), 2, 38, ["identifier"]),
                # "/": Function(lambda x, y, variableScope: devide(x, y), 2, 38, ["identifier"])}
    identifier = {}
    keyword = ["int", "float", "char", "String", "if", "ifn", "while", "func"]
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
            if currentHighest.parameters == 3:
                return ASTBranch(restFunctions[currentHighestIndex].code,
                                    (restFunctions[currentHighestIndex + 1],
                                    restFunctions[currentHighestIndex + 2],
                                    restFunctions[currentHighestIndex + 3],
                                    list(map(lambda x: environment.get(x) if (x in environment) else x, currentHighest.environment))))

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
            if (len(restFunctions) > 1):
                return restFunctions
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
        codeTextIndex = code.find(".text")
        codeDataIndex = code.find(".data")
        newCodeTextIndex = newCode.find(".text")
        newCodeDataIndex = newCode.find(".data")
        returnValue = ""

        if codeDataIndex != -1 or newCodeDataIndex != -1:
            returnValue += ".data\n"
    
        if codeDataIndex != -1:
            returnValue += code[codeDataIndex+6:codeTextIndex]
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"
        if newCodeDataIndex != -1:
            for lines in newCode[newCodeDataIndex+6:newCodeTextIndex].split('\n'):
                if not lines in returnValue:
                    returnValue += lines + "\n"
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"
        
        if len(returnValue) > 0 and returnValue[-2:] != "\n\n":
            if returnValue[-1] != "\n":
                returnValue += "\n"
            returnValue += "\n"
        
        if codeTextIndex != -1 or newCodeTextIndex != -1:
            returnValue += ".text\n"
        if codeTextIndex != -1:
            returnValue += code[codeTextIndex+6:]
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"
            # print("3 " + code[codeTextIndex+6:])
            # print("----")
        if newCodeTextIndex != -1:
            returnValue += newCode[newCodeTextIndex+6:]
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"
            # print("4 " + newCode[newCodeTextIndex+6:])
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
    startText = ".text\n\t.global _start\n_start:\n"
    endText = ".text\n\tmov "+ architect.R0 + ", #1"
    # endText = ".text\n\tmov "+ architect.R0 + ", #1\n\tint 0x80"
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

# print(output[0])

outputFile = open("out.asm", "w")
outputFile.write(output[0])