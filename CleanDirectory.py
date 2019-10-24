import os
from PIL import Image

''' Not used yet -------------------------------------------------------------------------------------------'''

# class to list out sub-directories of a directory
class List_Directories:
    def __init__(self, rootDir, sub_directory):
        self.rootDir = rootDir
        self.sub_directory = sub_directory
        self.dir_list = []
        dir = str(rootDir )
        if os.path.exists(dir):
            self.dirFile = open(dir + '/' + 'subdirListing.txt', 'w')
            for dirName, subdirList, fileList in os.walk(self.rootDir + '/' + 'train'):
                print('Found directory: %s' % dirName)
                self.dir_list.append(dirName)
            #for dirs in dirName:
                self.dirFile.write(str(dirName) + ',')
            self.dirFile.close()
        else:
            print('file does not exist')

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