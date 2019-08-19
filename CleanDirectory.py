import os
from PIL import Image

# clean directories
class CleanDirectories:
    global testFile
    def __init__(self, rootDir):
        self.rootDir = rootDir
        for dirName, subdirList, fileList in os.walk(self.rootDir):
            print('Found directory: %s' % dirName)
            for filename in fileList:
                print('\t%s' % filename)
                self.testFile = dirName + '/' + filename

                try:
                    im = Image.open(self.testFile)
                    im.verify() #I perform also verify, don't know if he sees other types o defects
                    im.close() #reload is necessary in my case
                    print("passed ",self.testFile)
                    #im = Image.load(filename)
                    #im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                    #im.close()

                except:
                    os.remove(self.testFile)
                    print("removed ",self.testFile)

if __name__ == "__main__":
    CleanDirectories("C:/Users/phfro/Documents/python/data/BirdiesData/train")
    #  CleanDirectories("F:/BirdiesData/train")