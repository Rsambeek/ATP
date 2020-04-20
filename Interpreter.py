from copy import deepcopy

inputFile = open("codeFile.bimsux", "r")
inputCode = ""
[(((inputCode += char) for char in word) for word in line) for line in inputCode]
print(inputCode)


class Token:
    name = ""
    tokenType = ""

    def __init__(self, name: str = "", tokenType: str = ""):
        self.name = name
        self.tokenType = tokenType

    def __repr__(self):
        return "<class 'Token_" + self.name + ":" + self.tokenType + "'>"

    def __str__(self):
        return self.name + " : " + self.tokenType

    def clear(self):
        self.name = ""
        self.tokenType = ""


class Variable:
    name: str
    dataType: type
    value: type

    def __init__(self, name: str = "", dataType: type = None, value: type = None):
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


class Interpreter:
    def __init__(self):
        self.tokenTypes = {"identifier": {},
                           "keyword": {"int": int, "float": float, "char": chr, "String": str},
                           "separator": [],
                           "operator": ["="],
                           # "literal": [],
                           # "comment": []
                           }
        # self.variables = {}
        self.brackets = []
        self.tokens = []
        self.error = ""

    def defineVariable(self, keywordToken, identifierToken, identifierList) -> str:
        if identifierToken.name in identifierList:
            return identifierToken.name + " is already defined"
        else:
            identifierList[identifierToken.name] = Variable(identifierToken.name, self.tokenTypes["keyword"][keywordToken.name])
            return

    def assignVariable(self, identifierToken, valueToken, identifierList) -> str:
        if identifierToken.name in self.tokenTypes["identifier"]:
            if valueToken.tokenType == "literal":
                identifierList[identifierToken.name].value = identifierList[identifierToken.name].dataType(valueToken.name)
            elif valueToken.tokenType == "identifier":
                identifierList[identifierToken.name].value = identifierList[identifierToken.name].dataType(identifierList[valueToken.name].value)

            return
        else:
            return "bad assignment"

    def scanner(self, inputCode, tokens, currentToken=Token()):
        if len(inputCode) > 0:
            if inputCode[0] == " ":
                if len(currentToken.name) != 0:
                    tokens.append(deepcopy(currentToken))
                    currentToken.clear()

            elif inputCode[0] == ";":
                tokens.append(deepcopy(currentToken))
                currentToken.clear()

            # elif line[0] == "\n":
            #     print("UHM excuse me da fuck, this is not python. Use ; at the end of the line, like a civil programmer.")
            #     return False

            elif inputCode[0] == "'" or inputCode[0] == '"':
                stringing = True
                self.brackets.append(inputCode[0])
                currentToken.tokenType = "literal"

            elif inputCode[0] == "=":
                if tokens[-1].name[0] == "=":
                    tokens[-1].name += "="
                else:
                    if currentToken.name != "":
                        tokens.append(deepcopy(currentToken))
                        currentToken.clear()
                    tokens.append(Token("=", "operator"))

            elif inputCode[0].isalpha():
                if currentToken.name == "" or currentToken.name[-1].isalpha():
                    currentToken.name += inputCode[0]
                else:
                    print("its me alpha")
                    return False

            elif inputCode[0].isdigit():
                if currentToken.name == "" or currentToken.name[-1].isdigit():
                    currentToken.name += inputCode[0]
                    if currentToken.tokenType == "":
                        currentToken.tokenType = "literal"
                else:
                    print("its me digit")
                    return False

            self.scanner(inputCode[1:], tokens, currentToken)

    def evaluator(self, tokens) -> None:
        if len(tokens) > 0:
            if tokens[0].name != "":
                if tokens[0].tokenType == "":
                    if tokens[0].name in self.tokenTypes["keyword"]:
                        tokens[0].tokenType = "keyword"
                    elif tokens[0].name in self.tokenTypes["separator"]:
                        tokens[0].tokenType = "separator"
                    elif tokens[0].name in self.tokenTypes["operator"]:
                        tokens[0].tokenType = "operator"

                    else:
                        tokens[0].tokenType = "identifier"

                self.evaluator(tokens[1:])

    def tokenizeCode(self, inputCode):
        self.scanner(inputCode, self.tokens)

        self.evaluator(self.tokens)

        tokenIndex = 0
        while tokenIndex < len(self.tokens):
            if self.tokens[tokenIndex].tokenType == "keyword":
                self.defineVariable(self.tokens[tokenIndex], self.tokens[tokenIndex + 1], self.tokenTypes["identifier"])
                tokenIndex += 1

            elif self.tokens[tokenIndex].tokenType == "operator":
                if tokenIndex != 0 and len(self.tokens) > tokenIndex + 1:
                    self.tokens[tokenIndex + 1]
                    self.assignVariable(self.tokens[tokenIndex - 1], self.tokens[tokenIndex + 1], self.tokenTypes["identifier"])
                    tokenIndex += 1
                else:
                    self.error += "unexpected operator"

            else:
                print("Failed recognizing token: " + self.tokens[tokenIndex].name)
                return False
            tokenIndex += 1

        print(self.tokens)
        print(self.tokenTypes["identifier"])
        for variable in self.tokenTypes["identifier"]:
            print(self.tokenTypes["identifier"][variable].name, self.tokenTypes["identifier"][variable].value, end=" | ")
        print(self.error)
        return True

interpreter = Interpreter()
if not interpreter.tokenizeCode(inputCode):
    print("Something went wrong")
