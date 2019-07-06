import os
from string import ascii_lowercase

class Categorize:
    def __init__(self, rootDir,targetDir):
        self.rootDir = rootDir
        self.targetDir = targetDir
        self.typeNames = []
        #  self.catType = catType
        for dirPath, dirNames, fileNames in os.walk(self.rootDir):
            #  print(dirNames)
            #  print("length dirNames= ",len(dirNames))
            counter = 0
            while counter < (len(dirNames)-1):
                for j in range(9):
                    #  print(dirNames[counter])
                    dirName = dirNames[counter].split()
                    #  print(type(dirName),"  ",len(dirName),"  ",dirName)
                    birdType = dirName[len(dirName)-1]
                    print(birdType)
                    if not ( os.path.isdir(self.targetDir + '/' + birdType)):
                    #    print(self.targetDir + '/' + birdType )
                        os.mkdir(self.targetDir + '/' + birdType )
                    if not ( os.path.isdir(self.targetDir + '/' + birdType + '/train')):
                    #    print(self.targetDir + '/' + birdType )
                        os.mkdir(self.targetDir + '/' + birdType + '/train')
                    counter = counter + 1
            #  while counter < (len(dirNames)-1):
                #  select the last name of the directory

                #  browse the list of target directories and locate the one with the same last name

                #  check each file and copy it if it doesnt already exist




if __name__ == "__main__":
    Categorize('F:/birdiesdata/train','F:/birdiesdata')