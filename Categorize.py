import os
from string import ascii_lowercase

class Categorize:
    def __init__(self, rootDir,targetDir):
        self.rootDir = rootDir
        self.targetDir = targetDir
        self.typeName = []
        #  self.catType = catType
        for dirPath, dirNames, fileNames in os.walk(self.rootDir):
            for dirN in dirNames:
                print(dirN)

                self.typeName.append(dirN.split(dirN)[0])
        print(self.typeName)
            #for letter in ascii_lowercase:
            #    if not ( os.path.isdir(self.targetDir + '/' + letter)):
            #        print(self.targetDir + '/' + letter)
            #        os.mkdir(os.path.isdir(self.targetDir + '/' + letter))


if __name__ == "__main__":
    Categorize('D:/birdiesdata/train','D:/birdiesdata/train')