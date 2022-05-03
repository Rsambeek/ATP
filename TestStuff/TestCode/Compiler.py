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
# Depricated class, used for storing data in interpreter
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
                    if (arg.tokenType == "literal"):    # If argument is literal, move to register
                        returnValue += "\tmov " + architect.REGISTERS[index] + ", #" + arg.name + "\n"
                    elif (arg.tokenType == "identifier"):   # If argument is identifier, fetch data and move to register
                        returnValue += "\tldr " + architect.R4 + ", =" + arg.name + "\n"
                        returnValue += "\tldr " + architect.REGISTERS[index] + ", [" + architect.R4 + "]\n"
                elif type(arg) == ASTBranch:    # If argument is ASTBranch, compile and insert assembly
                    returnValue = asmMerger(returnValue, compileASTBranch(arg))
                index += 1
            returnedValue = func(*args, **kwargs)
            
            # Merge function assembly with outer assembly
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
        output = ".text\n"
        if type(input) == str:
            index = input.find(".text\n")
            if index == -1:
                output = input + ".text\n"
        return output

    # assignVariable :: Token -> Token -> Token
    def assignVariable(token1: Token, token2: Token) -> Token:
        if type(token1) != Token:
            raise Exception("token1 is not Token")
        elif type(token2) != Token:
            raise Exception("token2 is not Token")
        elif token1.tokenType != "identifier":
            raise Exception(token1.name + " is not an identifier")
        
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
        if type(identifierToken) != Token:
            raise Exception("token1 is not Token")
        elif identifierToken.tokenType != "identifier":
            raise Exception(identifierToken.name + " is not an identifier")

        returnToken = Token(identifierToken.name)
        newAssembly = "\t" + identifierToken.name + ": .space 4\n"
        if type(identifierToken.operation) == str:
            index = identifierToken.operation.find(".data\n")
            if index != -1:
                returnToken.operation = identifierToken.operation[:index + 6] + newAssembly + identifierToken.operation[index + 6:]
            else:
                returnToken.operation = ".data\n" + newAssembly + identifierToken.operation
        else:
            returnToken.operation = ".data\n" + newAssembly

        return returnToken

    # functionHeader :: Token -> Union[List, Token, None] -> Union[ASTBranch, None] -> Token
    def functionHeader(token1: Token, parameters: Union[List, Token, None] = None, codeBlock: Union[ASTBranch, None] = None) -> Token:
        if type(token1) != Token:
            raise Exception("token1 is not Token")
        elif token1.tokenType != "identifier":
            raise Exception(token1.name + " is not an identifier")

        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)

        # set global function name, jump to after function
        returnToken.operation += "\n\t.global " + token1.name + "\n"
        returnToken.operation += "\tb _after" + token1.name + "\n"
        returnToken.operation += token1.name + ":\n"
        
        # push registers to stack
        returnToken.operation += "\tpush {"
        for register in architect.REGISTERS[architect.SAFEINDEX:]:
            returnToken.operation += register + ", "
        
        returnToken.operation += "lr}\n"

        if (type(parameters) == Token):
            parameters = [parameters]
        
        # Define parameters usable by function
        for i in range(len(parameters)):
            if parameters[i].name == "":
                continue

            newAsm = newInt(parameters[i]).operation
            if (returnToken.operation.find(newAsm) == -1):
                returnToken.operation = asmMerger(returnToken.operation, newAsm)
            
            # Push registers(parameters) to defined variables
            returnToken.operation += "\tldr " + architect.R4 + ", =" + parameters[i].name + "\n"
            returnToken.operation += "\tstr " + architect.REGISTERS[i] + ", [" + architect.R4 + "]\n"

        returnToken.operation = asmMerger(returnToken.operation, compileASTBranch(codeBlock).operation)
        
        # pop stacked back to registers
        returnToken.operation += "\tpop {"
        for register in architect.REGISTERS[architect.SAFEINDEX:]:
            returnToken.operation += register + ", "
        
        returnToken.operation += "PC}\n\n"
        returnToken.operation += "_after" + token1.name + ":\n"
        return returnToken
    
    # callFunction :: Token -> Union[List, Token, None] -> Token
    def callFunction(token1: Token, parameters: Union[List, Token, None] = None) -> Token:
        # Function for calling predefined function blocks
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        if (type(parameters) == Token):
            parameters = [parameters]

        for i in range(len(parameters)):
            if parameters[i].name == "":
                continue
            
            # Pull given variables to registers
            returnToken.operation += "\tldr " + architect.R4 + ", =" + parameters[i].name + "\n"
            returnToken.operation += "\tldr " + architect.REGISTERS[i] + ", [" + architect.R4 + "]\n"

        returnToken.operation += "\tbl " + token1.name + "\n"
        return returnToken
    
    # returnFunction :: Token -> Token
    def returnFunction(token1 : Token) -> Token:
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)

        # Return argument from function by storing it in register 0
        if (token1.tokenType == "literal"):     # If literal, store literal in register 0
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):    # If identifier, load value and store in register 0
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        else:   # Else merge returned assembly
            returnToken.operation = asmMerger(returnToken.operation, token1.operation)
        
        
        # pop stacked back to registers
        returnToken.operation += "\tpop {"
        for register in architect.REGISTERS[architect.SAFEINDEX:]:
            returnToken.operation += register + ", "
        
        returnToken.operation += "PC}\n\n"

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

        # Compile code to assembly
        returnedValue = compileASTBranch(codeBlock).operation
        
        # Create unique label from returned assembly for if statement
        functionHash = hash(returnedValue)
        functionHash += sys.maxsize + 1
        functionHash = token1.name + "_" + str(functionHash)

        if (token1.tokenType == "literal"):     # If literal, store literal in register 0
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):    # If identifier, load value and store in register 0
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        
        # Compare register 0, skip if register 0 is 0
        returnToken.operation += "\tcmp " + architect.R0 + ", #0\n\tbeq _afterif" + functionHash + "\n"
        returnToken.operation = asmMerger(returnToken.operation, returnedValue)
        returnToken.operation += "_afterif" + functionHash + ":\n\n"
        return returnToken

    # ifnStatement :: Token -> Union[ASTBranch, None] -> Token
    def ifnStatement(token1: Token, codeBlock: Union[ASTBranch, None] = None) -> Token:
        returnToken = Token(token1.name)
        returnToken.operation = ""
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "\n"

        # Compile code to assembly
        returnedValue = compileASTBranch(codeBlock).operation

        # Create unique label from returned assembly for if statement
        functionHash = hash(returnedValue)
        functionHash += sys.maxsize + 1
        functionHash = token1.name + "_" + str(functionHash)

        if (token1.tokenType == "literal"):     # If literal, store literal in register 0
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):    # If identifier, load value and store in register 0
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        
        # Compare register 0, skip if register 0 is not 0
        returnToken.operation += "\tcmp " + architect.R0 + ", #0\n\tbne _afterifn" + functionHash + "\n"
        returnToken.operation = asmMerger(returnToken.operation, returnedValue)
        returnToken.operation += "_afterifn" + functionHash + ":\n\n"
        return returnToken

    # whileStatement :: Token -> ASTBranch -> List[str]
    def whileStatement(token1: Token, codeBlock: ASTBranch) -> List[str]:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "\n"

        if (token1.tokenType == "literal"):     # If literal, store literal in register 0
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):    # If identifier, load value and store in register 0
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        
        # Check condition, if condition is false, jump to end of while loop
        returnToken.operation += "\tcmp " + architect.R0 + ", #0\n\tbeq _afterwhile" + token1.name + "\n_startwhile" + token1.name + ":\n\n"
        
        # Merge while operation with code block assembly
        returnToken.operation = asmMerger(returnToken.operation, compileASTBranch(codeBlock).operation)

        if (token1.tokenType == "literal"):
            returnToken.operation += "\tmov " + architect.R0 + ", #" + token1.name + "\n"
        elif (token1.tokenType == "identifier"):
            returnToken.operation += "\tldr " + architect.R1 + ", =" + token1.name + "\n"
            returnToken.operation += "\tldr " + architect.R0 + ", [" + architect.R1 + "]\n"
        
        # Check condition, if condition is not 0, jump to start of while loop
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
        
        returnToken.operation += "\tsub " + architect.R0 + ", " + architect.R1 + "\n"
        return returnToken

    # # multiply :: Token -> Token -> Token
    @registerPrepDecorator
    def multiply(token1: Token, token2: Token) -> Token:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "\tmul " + architect.R0 + ", " + architect.R1 + "\n"
        return returnToken

    # # devide :: Token -> Token -> Token
    @registerPrepDecorator
    def devide(token1: Token, token2: Token) -> Token:
        returnToken = Token(token1.name)
        if type(token1.operation) == str:
            returnToken.operation = token1.operation
        else:
            returnToken.operation = ""
        
        returnToken.operation = prepareAsmStatement(returnToken.operation)
        
        returnToken.operation += "\div " + architect.R0 + ", " + architect.R1 + "\n"
        return returnToken

    # Map of all defined operators
    operations = {"int": Function(lambda x, variableScope: newInt(x), 1, 10, ["identifier"]),
                "func": Function(lambda x, y, z, variableScope: functionHeader(x, y, z), 3, 10, ["identifier"]),
                "call": Function(lambda x, y, variableScope: callFunction(x, y), 2, 10, ["identifier"]),
                "return": Function(lambda x, variableScope: returnFunction(x), 1, 10, ["identifier"]),
                "if": Function(lambda x, y, variableScope: ifStatement(x, y), 2, 20, ["identifier"]),
                "ifn": Function(lambda x, y, variableScope: ifnStatement(x, y), 2, 20, ["identifier"]),
                "while": Function(lambda x, y, variableScope: whileStatement(x, y), 2, 20, ["identifier"]),
                "=": Function(lambda x, y, variableScope: assignVariable(x, y), 2, 80, ["identifier"]),
                "+": Function(lambda x, y, variableScope: add(x, y), 2, 39, ["identifier"]),
                "-": Function(lambda x, y, variableScope: subtract(x, y), 2, 39, ["identifier"]),
                "*": Function(lambda x, y, variableScope: multiply(x, y), 2, 38, ["identifier"]),
                "/": Function(lambda x, y, variableScope: devide(x, y), 2, 38, ["identifier"])}
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
        # Set token typing
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
    def lexer(inputCode: str, currentToken: Token = Token()) -> List[List[Token]]:
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
        # If token is a function return actual function
        if token.name in operations:
            return operations[token.name]
        else:
            return token

    # compareOrder :: Union [Function, Token, ASTBranch] -> Union [Function, Token, ASTBranch] -> Union[Function, Token, ASTBranch]
    def compareOrder(operation1: Union[Function, Token, ASTBranch], operation2: Union[Function, Token, ASTBranch]) -> Union[Function, Token, ASTBranch]:
        # Compare order of operations and return highest priority operation
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
                        # If current item is in bracket pair, then it is an opening bracket so add it to the stack
                        bracketStack.append(bracketPairs[currentItem])

                    elif bracketStack[-1] == currentItem:
                        # If current item is in bracket stack, then it is an closing bracket so remove it from the stack
                        bracketStack.pop()
                    else:
                        # if not an opening bracket and not the right closing bracket, raise exception
                        raise Exception("Bracket mismatch")

                if len(bracketStack) <= 0:
                    return [], restList[1:]

            if len(restList[1:]) <= 0:
                raise Exception("Bracket mismatch")

            temp = companionBracketFinder(restList[1:], separators, bracketStack)
            return [restList[0]] + temp[0], temp[1]

    # treeConstructor :: List[Union[Function, Token, ASTBranch]] -> dict -> Union[ASTBranch, Token]
    def treeConstructor(restFunctions: List[Union[Function, Token, ASTBranch]], environment: dict) -> Union[ASTBranch, Token]:
        if len(restFunctions) == 0:
            return Token("")
        currentHighest = reduce(compareOrder, restFunctions)
        currentHighestIndex = restFunctions.index(currentHighest)

        # If current highest is ASTBranch, then assess the rest of the list and return outcome
        if type(currentHighest) == ASTBranch:
            if (len(restFunctions) > 1):
                return ASTBranch(asmMerger, (currentHighest, treeConstructor(restFunctions[1:], [{}])))
            return currentHighest

        # If current highest is function, then fill parameters and return ASTBranch
        elif type(currentHighest) == Function:
            if currentHighest.parameters == 3:
                returnValue = ASTBranch(restFunctions[currentHighestIndex].code,
                                    (restFunctions[currentHighestIndex + 1],
                                    restFunctions[currentHighestIndex + 2],
                                    restFunctions[currentHighestIndex + 3],
                                    list(map(lambda x: environment.get(x) if (x in environment) else x, currentHighest.environment))))

            elif currentHighest.parameters == 2:
                if 0 < currentHighestIndex < len(restFunctions):
                    return ASTBranch(restFunctions[currentHighestIndex].code,
                                     (treeConstructor(restFunctions[:currentHighestIndex], environment),
                                      treeConstructor(restFunctions[currentHighestIndex+1:], environment),
                                      list(map(environment.get, currentHighest.environment))))
                else:
                    returnValue = ASTBranch(restFunctions[currentHighestIndex].code,
                                     (restFunctions[currentHighestIndex + 1],
                                      restFunctions[currentHighestIndex + 2],
                                      list(map(lambda x: environment.get(x) if (x in environment) else x, currentHighest.environment))))

            else:
                if currentHighestIndex != 0:
                    print("Unexpected Operation")
                returnValue = ASTBranch(restFunctions[currentHighestIndex].code,
                                (restFunctions[currentHighestIndex + 1],
                                  list(map(environment.get, currentHighest.environment))))

            if currentHighest.parameters < len(restFunctions) - 1:
                return treeConstructor((returnValue, treeConstructor(restFunctions[(currentHighestIndex + currentHighest.parameters + 1):], environment)), environment)
            else:
                return returnValue

        # If current highest is seperator, find companion bracket and assess that branch
        elif currentHighest.tokenType == "separator":
            temp = companionBracketFinder(restFunctions[currentHighestIndex:],environment["separator"])
            restFunctions = restFunctions[:currentHighestIndex] + [treeConstructor(temp[0], environment)] + temp[1]
            return treeConstructor(restFunctions, environment)

        else:
            if (len(restFunctions) > 1):
                return restFunctions
            else:
                return currentHighest
            # return currentHighest

    # parser :: List[List[Token]] -> dict -> List[Union[ASTBranch, Token]]
    def parser(tokens: List[List[Token]], environment: dict) -> List[Union[ASTBranch, Token]]:
        functions = list(map(lambda innerList: list(map(lambda item: functionalGetter(environment["operations"], item), innerList)), tokens))
        AST = list(map(lambda currentExpression: treeConstructor(currentExpression, environment), functions))

        return AST

    # compileASTBranch :: Union[ASTBranch, Token] -> str
    def compileASTBranch(input: Union[ASTBranch, Token]) -> str:
        # if type(input == list):
            # print(type(input), end=" | ")
            # print(input)
        if type(input) == Token:
            if input.operation == None:
                input.operation = input.name
            return input

        elif type(input) == ASTBranch:
            # Asses is code should be run or is run by calling function
            # Deprecated, only used in interpreter
            if type(input.arguments[-1]) == list and "noRun" in input.arguments[-1]:
                input.arguments[-1].pop(input.arguments[-1].index("noRun"))
                return input.function(*input.arguments)

            else:
                return input.function(*list(map(compileASTBranch, input.arguments)))

        else:
            return input
    
    # asmMerger :: Union[str, Token] -> Union[str, Token] -> Union[str, Token]
    def asmMerger(code: Union[str, Token], newCode: Union[str, Token]) -> Union[str, Token]:
        returnToken = False
        if type(code) == Token:
            code = code.operation
            returnToken = True

        codeTextIndex = code.find(".text")
        codeDataIndex = code.find(".data")
        
        if type(newCode) == Token:
            newCode = newCode.operation

        newCodeTextIndex = newCode.find(".text")
        newCodeDataIndex = newCode.find(".data")
        
        returnValue = ""

        # Prepare data section
        if codeDataIndex != -1 or newCodeDataIndex != -1:
            returnValue += ".data\n"
    
        # Cut out data part and set in new code
        if codeDataIndex != -1:
            returnValue += code[codeDataIndex+6:codeTextIndex]
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"

        # Cut out new data part and set in new code
        if newCodeDataIndex != -1:
            for lines in newCode[newCodeDataIndex+6:newCodeTextIndex].split('\n'):
                if not lines in returnValue:
                    returnValue += lines + "\n"
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"
        
        # Prepare text section
        if len(returnValue) > 0 and returnValue[-2:] != "\n\n":
            if returnValue[-1] != "\n":
                returnValue += "\n"
            returnValue += "\n"
        
        # Cut out text part and set in new code
        if codeTextIndex != -1 or newCodeTextIndex != -1:
            returnValue += ".text\n"
        if codeTextIndex != -1:
            returnValue += code[codeTextIndex+6:]
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"
        
        # Cut out new text part and set in new code
        if newCodeTextIndex != -1:
            returnValue += newCode[newCodeTextIndex+6:]
            while returnValue[-1] == "\n":
                returnValue = returnValue[:-1]
            returnValue += "\n"
        
        if returnToken:
            return Token(returnValue, returnValue)
        return returnValue

    # compile:: List[ASTBranch] -> Tuple[str, bool]
    def compile(functions: List[ASTBranch]) -> Tuple[str, bool]:
        if len(functions) <= 0:
            output = ""
            return output, False
        else:
            newOutput = compileASTBranch(functions[0])
            if type(newOutput) == Token:
                if newOutput.operation != None:
                    newOutput = newOutput.operation
                else:
                    newOutput = newOutput.name
            output = compile(functions[1:])
            if type(newOutput) == list:
                return newOutput + output[0], output[1]
            else:
                newOutput = asmMerger(newOutput, output[0])
                return newOutput, output[1]


    tokens = lexer(inputCode)
    AST = parser(tokens, environment)
    output = compile(AST)
    startText = ".text\n\t.global _start\n_start:\n"
    endText = ".text\n\tmov "+ architect.R0 + ", #1"
    outputText = asmMerger(startText, output[0])
    outputText = asmMerger(outputText, endText)
    output = outputText, output[1]


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