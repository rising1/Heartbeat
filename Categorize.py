import os
from string import ascii_lowercase
from PIL import Image

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
        ''' This part creates all the necessary directory names  -----------------------------'''
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
                print("directory creation complete")
                ''' End of part creates all the necessary directory names  --------------------'''
                #  Now walk the target directory
                # exclude = set(["train","val","test"])
                #  for tdirPath, tdirNames, tfileNames in os.walk(self.targetDir):
                #  if len(tdirNames) > 3:
                #  tdirNames[:] = [d for d in tdirNames if d not in exclude]
                for tdirNames in next(os.walk(self.targetDir))[1] :
                    for sdirNames in next(os.walk(self.rootDir))[1]:
                        sdirName = sdirNames.split()
                        lastsdirName = sdirName[len(sdirName)-1]
                        if tdirNames == lastsdirName:
                            print("found ",sdirName, " to copy to ", tdirNames)
                            for sfileNames in next(os.walk(self.rootDir + '/' + sdirNames))[2]:
                                print("sfileNames=",sfileNames)
                                #  check file
                                try:
                                    im = Image.open(self.rootDir + '/' + sdirNames + '/' + sfileNames)
                                    im.verify()  # I perform also verify, don't know if he sees other types o defects
                                    im.close()  # reload is necessary in my case
                                    # im = Image.load(filename)
                                    # im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                                    # im.close()
                                    print("sfileNames=", sfileNames, " -passed")

                                except:
                                    print("sfileNames=", sfileNames, " -failed")
                                #    os.remove(self.testFile)
                                #  copy file name with prefix sdirNames

                                #  if file doesnt exist

                                # copy file into directory







if __name__ == "__main__":
    Categorize('d:/birdiesdata/train','d:/birdiesdata')
    #  Categorize('h:/birdiesdata/train', 'h:/birdiesdata')