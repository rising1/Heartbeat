import os
from string import ascii_lowercase

class Categorize:
    def __init__(self, rootDir,targetDir):
        self.rootDir = rootDir
        self.targetDir = targetDir
        self.typeNames = []
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
                    if birdType not in self.typeNames:
                        self.typeNames.append(birdType)
                    counter = counter + 1
        self.create_dir()
        print("length of birdType= ",len(self.typeNames))

    def create_dir(self):
            for tname in self.typeNames:
               if not ( os.path.isdir(self.targetDir + '/' + tname)):
                    print(self.targetDir + '/' + tname + '/train/)
            #        os.mkdir(os.path.isdir(self.targetDir + '/' + letter + '/train/))


if __name__ == "__main__":
    Categorize('H:/birdiesdata/train','H:/birdiesdata')