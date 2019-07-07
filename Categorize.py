import os
from string import ascii_lowercase

class Categorize:
    def __init__(self, rootDir,targetDir):
        self.rootDir = rootDir
        self.targetDir = targetDir
        self.typeNames = []
        self.counter = 0
        #  self.catType = catType
        for dirPath, dirNames, fileNames in os.walk(self.rootDir):
            #  print(dirNames)
            print("length dirNames= ",len(dirNames))
            if len(dirNames)>0:
                while self.counter < (len(dirNames)-1):
                    #  print(dirNames[counter])
                    dirName = dirNames[self.counter].split()
                    #  print(type(dirName),"  ",len(dirName),"  ",dirName)
                    birdType = dirName[len(dirName)-1]
                    print(birdType)
                    if not ( os.path.isdir(self.targetDir + '/' + birdType)):
                        #    print(self.targetDir + '/' + birdType )
                        os.mkdir(self.targetDir + '/' + birdType )
                    if not ( os.path.isdir(self.targetDir + '/' + birdType + '/train')):
                        #    print(self.targetDir + '/' + birdType )
                        os.mkdir(self.targetDir + '/' + birdType + '/train')
                    self.counter = self.counter + 1
                #  Now walk the target directory
                for tdirPath, tdirNames, tfileNames in os.walk(self.targetDir):
                #  If the name of the target directory = last name of the source directory

                #       Then iterate through the source directory

                #           pick out each source file, prefix it with iteration no,
                #           and test if exists in the target

                #               if not exists in target then copy source file into the target

                #               Dont touch the train directory






if __name__ == "__main__":
    Categorize('F:/birdiesdata/train','F:/birdiesdata')