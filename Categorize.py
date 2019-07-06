import os
from string import ascii_lowercase

class Categorize:
    def __init__(self, rootDir,targetDir):
        self.rootDir = rootDir
        self.targetDir = targetDir
        self.typeName = []
        #  self.catType = catType
        for dirPath, dirNames, fileNames in os.walk(self.rootDir):
            print(dirNames)
            print("length dirNames= ",len(dirNames))
            counter = 0
            while counter < (len(dirNames)-1):
                for j in range(9):
                    #  print(dirNames[counter])
                    dirName = dirNames[counter].split()
                    #  print(type(dirName),"  ",len(dirName),"  ",dirName)
                    birdType = dirName[len(dirName)-1]
                    print(birdType)
                    if birdType not in self.typeName:
                        self.typeName.append(birdType)
                    counter = counter + 1
        print(self.typeName)
        print("length of birdType= ",len(self.typeName))
            #  for letter in ascii_lowercase:
            #    if not ( os.path.isdir(self.targetDir + '/' + letter)):
            #        print(self.targetDir + '/' + letter)
            #        os.mkdir(os.path.isdir(self.targetDir + '/' + letter))


if __name__ == "__main__":
    Categorize('H:/birdiesdata/train','H:/birdiesdata/train')