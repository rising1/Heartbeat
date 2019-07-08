import os
from string import ascii_lowercase

class Categorize:
    def __init__(self, rootDir,targetDir):
        self.rootDir = rootDir
        self.targetDir = targetDir
        self.typeNames = []
        self.counter = 0
        self.tcounter = 0
        self.scounter = 0
        self.birdType = ""
        self.sbirdType = ""
        #  self.catType = catType
        for dirPath, dirNames, fileNames in os.walk(self.rootDir):
            #  print(dirNames)
            #  print("length dirNames= ",len(dirNames))
            if len(dirNames)>0:
                print("length dirNames= ", len(dirNames))
                while self.counter < (len(dirNames)-1):
                    #  print(dirNames[counter])
                    dirName = dirNames[self.counter].split()
                    #  print(type(dirName),"  ",len(dirName),"  ",dirName)
                    self.birdType = dirName[len(dirName)-1]
                    #  print(self.birdType)
                    if not ( os.path.isdir(self.targetDir + '/' + self.birdType)):
                        #    print(self.targetDir + '/' + birdType )
                        os.mkdir(self.targetDir + '/' + self.birdType )
                    if not ( os.path.isdir(self.targetDir + '/' + self.birdType + '/train')):
                        #    print(self.targetDir + '/' + birdType )
                        os.mkdir(self.targetDir + '/' + self.birdType + '/train')
                    self.counter = self.counter + 1
                #  Now walk the target directory
                for tdirPath, tdirNames, tfileNames in os.walk(self.targetDir):
                    if len(tdirNames) > 3:
                        print("length tdirNames= ", len(tdirNames))
                #    for sdirPath, sdirNames, sfileNames in os.walk(self.rootDir):
                #        print("length tdirNames= ", len(tdirNames))
                #        while self.tcounter < (len(tdirNames) - 1):
                #            if len(sdirNames) > 0:
                #                print("length sdirNames= ", len(sdirNames))
                #                while self.scounter < (len(sdirNames) - 1):
                #                    sdirName = sdirNames[self.scounter].split()
                #                    self.sbirdType = sdirName[len(sdirName) - 1]
                #                    if self.sbirdType == tdirNames[self.tcounter]:
                #                        print("tcounter=",self.sbirdType)
                #                    self.scounter = self.scounter + 1
                #            self.tcounter = self.tcounter + 1
                #  If the name of the target directory = last name of the source directory

                #       Then iterate through the source directory

                #           pick out each source file, prefix it with iteration no,
                #           and test if exists in the target

                #               if not exists in target then copy source file into the target

                #               Dont touch the train directory






if __name__ == "__main__":
    Categorize('d:/birdiesdata/train','d:/birdiesdata')