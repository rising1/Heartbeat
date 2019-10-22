import os
from PIL import Image

class List_Directories:
    def __init__(self, rootDir, sub_directory):
        self.rootDir = rootDir
        self.sub_directory = sub_directory
        self.dirFile = open(rootDir + 'dirFile', 'rw+')
        for dirName, subdirList, fileList in os.walk(self.rootDir):
            print('Found directory: %s' % dirName)
            for subdir in subdirList:
                self.dirFile.write(subdir + ',')
            self.dirFile.close()

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
    # CleanDirectories("C:/Users/phfro/Documents/python/data/BirdiesData/train")
    CleanDirectories('C:/Users/phfro/PycharmProjects/Heartbeat/train')
    CleanDirectories('C:/Users/phfro/PycharmProjects/Heartbeat/val')
    CleanDirectories('C:/Users/phfro/PycharmProjects/Heartbeat/test')
    #  CleanDirectories("F:/BirdiesData/train")