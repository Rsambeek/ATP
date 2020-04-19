from enum import Enum

inputCode = open("codeFile.penis", "r")


class Interpreter:
    def __init__(self):
        self.tokenTypes = {"identifier": {},
                           "keyword": {"if": [], "while": [], "return": [], "int": int, "float": float, "char": chr, "String": str},
                           "separator": [],
                           "operator": ["=", "+"],
                           "literal": [],
                           "comment": []}
        self.variables = {}
        self.brackets = []
        self.tokens = []
        self.error = ""

    def defineVariable(self, keywordToken, identifierToken) -> None:
        if identifierToken[0] in self.tokenTypes["identifier"]:
            self.error = identifierToken[0] + " is already defined"
            return
        else:
            self.tokenTypes["identifier"][identifierToken[0]] = self.tokenTypes["keyword"][keywordToken[0]]
            self.variables[identifierToken[0]] = self.tokenTypes["identifier"][identifierToken[0]]()
            return

    def assignVariable(self, identifierToken, valueToken) -> None:
        if identifierToken[0] in self.tokenTypes["identifier"]:
            if valueToken[1] == "literal":
                self.variables[identifierToken[0]] = self.tokenTypes["identifier"][identifierToken[0]](valueToken[0])
                return
            elif valueToken[1] == "identifier":
                self.variables[identifierToken[0]] = self.tokenTypes["identifier"][identifierToken[0]](self.variables[valueToken[0]])
                return

        else:
            self.error = "bad assignment"
            return

    def tokenizeCode(self, inputCode):
        currentToken = [""]

        def tokenize():
            nonlocal currentToken
            if len(currentToken) == 1 and len(currentToken[0]) > 0:
                if currentToken[0] in self.tokenTypes["keyword"]:
                    currentToken.append("keyword")
                elif currentToken[0] in self.tokenTypes["separator"]:
                    currentToken.append("separator")
                elif currentToken[0] in self.tokenTypes["operator"]:
                    currentToken.append("operator")

                else:
                    currentToken.append("identifier")

            self.tokens.append(currentToken)
            currentToken = [""]

        for line in inputCode:
            stringing = False
            while len(line) > 0:
                if stringing:
                    if line[0] == self.brackets[-1]:
                        stringing = False
                        self.brackets.pop()
                        tokenize()
                    else:
                        currentToken[0] += line[0]

                else:
                    if line[0] == " ":
                        if len(currentToken[0]) != 0:
                            tokenize()

                    elif line[0] == ";":
                        tokenize()

                    # elif line[0] == "\n":
                    #     print("UHM excuse me da fuck, this is not python. Use ; at the end of the line, like a civil programmer.")
                    #     return False

                    elif line[0] == "'" or line[0] == '"':
                        stringing = True
                        currentToken.append("literal")

                    elif line[0] == "=":
                        if self.tokens[-1][0][0] == "=":
                            self.tokens[-1][0] += "="
                        else:
                            if len(currentToken[0]) != 0:
                                tokenize()
                            self.tokens.append(["=", "operator"])

                    elif line[0].isalpha():
                        if (len(currentToken[0]) == 0) or currentToken[-1].isalpha():
                            currentToken[0] += line[0]
                        else:
                            print("its me alpha")
                            return False

                    elif line[0].isdigit():
                        if (len(currentToken[0]) == 0) or currentToken[0][-1].isdigit():
                            currentToken[0] += line[0]
                            if len(currentToken) == 1:
                                currentToken.append("literal")
                        else:
                            print("its me digit")
                            return False

                line = line[1:]

        tokenIndex = 0
        while tokenIndex < len(self.tokens):
            if self.tokens[tokenIndex][1] == "keyword":
                self.defineVariable(self.tokens[tokenIndex], self.tokens[tokenIndex + 1])
                tokenIndex += 1

            elif self.tokens[tokenIndex][1] == "operator":
                self.assignVariable(self.tokens[tokenIndex - 1], self.tokens[tokenIndex + 1])
                tokenIndex += 1

            # elif self.tokens[tokenIndex][1] == "identifier":
            #     self.

            # elif (tokens[index] == "=" and tokens[index-1] != None):
            else:
                print("Failed recognizing token: " + self.tokens[0][0])
                return False
            tokenIndex += 1
        print(self.tokenTypes["identifier"])
        print(self.tokens)
        print(self.variables)
        return True

interpreter = Interpreter()
if not interpreter.tokenizeCode(inputCode) :
    print("Something went wrong")
