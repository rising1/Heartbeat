import os
from string import ascii_lowercase
from PIL import Image
from shutil import copyfile

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
                #  print("length dirNames= ", len(dirNames))
                while self.counter < (len(dirNames)):
                    #  print(dirNames[counter])
                    dirName = dirNames[self.counter].split()
                    #  print(type(dirName),"  ",len(dirName),"  ",dirName)
                    self.birdType = dirName[len(dirName)-1]
                    #  print(self.targetDir + '/' + self.birdType)
                    if not ( os.path.isdir(self.targetDir + '/' + self.birdType)):
                        #  print(self.targetDir + '/' + self.birdType )
                        os.mkdir(self.targetDir + '/' + self.birdType )
                    #  if not ( os.path.isdir(self.targetDir + '/' + self.birdType + '/train')):
                        #    print(self.targetDir + '/' + birdType )
                    #      os.mkdir(self.targetDir + '/' + self.birdType + '/train')
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
                        firstdirName = sdirName[0]
                        if tdirNames == lastsdirName:
                            print("found ",sdirName, " to copy to ", tdirNames)
                            for sfileNames in next(os.walk(self.rootDir + '/' + sdirNames))[2]:
                                print("sfileNames=",sfileNames)
                                #  check file
                                try:
                                    print("sfileNames=", sfileNames, " -passed")
                                    sourcepath_name = self.rootDir + '/' + sdirNames + '/' + sfileNames
                                    filepath_name = self.targetDir + '/' + tdirNames + '/train/' \
                                                    + firstdirName + '_' + sfileNames
                                    print("filepath_name= ",filepath_name)
                                    exists =  os.path.isfile(filepath_name)
                                    print("exists=",exists)
                                    if not exists:
                                        print("trying to open ",sourcepath_name)
                                        im = Image.open(sourcepath_name)
                                        im.verify()  # I perform also verify, don't know if he sees other types o defects
                                        im.close()  # reload is necessary in my case
                                        copyfile(sourcepath_name,filepath_name)
                                except Exception as e:
                                    print(e," --sfileNames=", sfileNames, " -failed")
                                #    os.remove(self.testFile)


if __name__ == "__main__":
    #  Categorize('d:/birdiesTest/train','d:/birdiesTest')
  Categorize('d:/birdiesdata/train','d:/birdiesdata2')
    #  Categorize('h:/birdiesdata/train', 'h:/birdiesdata')