import os
from string import ascii_lowercase

class Categorize:
    def __init__(self, rootDir,targetDir):
        self.rootDir = rootDir
        self.targetDir = targetDir
        self.catType = catType
        for dirPath, dirNames, fileNames in os.walk(self.rootDir):
            print(dirNames)
            for letter in ascii_lowercase:
                if not ( os.path.isdir(self.targetDir + '/' + letter)):
                    print(self.targetDir + '/' + letter)
                    #  os.mkdir(os.path.isdir(self.targetDir + '/' + letter))


if __name__ == "__main__":
    Categorize('root directory','target directory','alphabetic')