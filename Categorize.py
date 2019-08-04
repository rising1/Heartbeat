import os
from string import ascii_lowercase
from PIL import Image
from shutil import copyfile
import ImageType

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
    def build_directories(self,parent):
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
                    if not ( os.path.isdir(self.targetDir + '/' + parent + '/' + self.birdType)):
                        #  print(self.targetDir + '/' + self.birdType )
                        os.mkdir(self.targetDir + '/' + parent + '/' + self.birdType )
                    #  if not ( os.path.isdir(self.targetDir + '/' + self.birdType + '/train')):
                        #    print(self.targetDir + '/' + birdType )
                    #      os.mkdir(self.targetDir + '/' + self.birdType + '/train')
                    self.counter = self.counter + 1
                print("directory creation complete")
                ''' End of part creates all the necessary directory names  --------------------'''

    def copy_and_clean(self):
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
                                    filepath_name = self.targetDir + '/' + tdirNames + '/'  \
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
    def create_test(self,parent,number):
        #  walk the target directory and create a list of the directory names and save into a file
        class_file = open(self.targetDir + "/" + "Class_validate.txt", "w")
        for tdirNames in next(os.walk(self.targetDir + "/train"))[1]:
            counter = 0
            #  print("tdirNames=",tdirNames)
            for tfileNames in next(os.walk(self.targetDir + "/train" + "/" \
                                           + tdirNames))[2]:
                first_image = tfileNames
                sourcepath_name = self.targetDir + "/train" + \
                                     "/" + tdirNames + "/" + \
                                     first_image
                filepath_name = self.targetDir + "/" + parent + \
                                     "/" + tdirNames + "/" + \
                                     first_image
                if counter == number:
                    print("sourcepath_name=",sourcepath_name)
                    print("filepath_name=",filepath_name)
                    source_exists = os.path.isfile(sourcepath_name)
                    target_exists = os.path.isfile(filepath_name)
                    if source_exists and not target_exists:
                        copyfile(sourcepath_name, filepath_name)
                counter = counter + 1
            class_file.write(tdirNames + ",")
        class_file.close()
        with open(self.targetDir + "/" + "Class_validate.txt", 'rb+')\
                as filehandle:
            filehandle.seek(-1, os.SEEK_END)
            filehandle.truncate()
            filehandle.close()
        #  Copy the first item from each directory into the val directory


    def summarise(self):

        for tdirNames in next(os.walk(self.targetDir))[1]:
            counter = 0
            #  print("tdirNames=",tdirNames)
            for tfileNames in next(os.walk(self.targetDir  + "/" \
                                           + tdirNames))[2]:
                counter = counter +1
            print(tdirNames + "\t" + str(counter))

    def is_it_a_bird(self):
        index_file = "scan_results.txt"
        indexpath = os.path.join(self.rootDir, index_file)
        print("index_path= ",indexpath)
        if os.path.exists(indexpath):
            os.remove(indexpath)
        imageType = ImageType.ImageType()
        with open(indexpath, "w") as f:
            for sdirNames in next(os.walk(self.rootDir))[1]:
                for sfileNames in next(os.walk(self.rootDir + '/' + sdirNames))[2]:
                    file_path = self.rootDir  + '/' + sdirNames + '/' + sfileNames
                    print(file_path)
                    if os.path.isfile(file_path):
                        try:
                            result = imageType.predict_image(file_path)
                        except RuntimeError:
                            result = "PREDICT_FAILURE"
                        print(result + "\t" + file_path)
                        f.write(result + "\t"  + "\t" + file_path + "\n" )
                        f.flush()
        f.close()


if __name__ == "__main__":
    #  Categorize('d:/birdiesTest/train','d:/birdiesTest')
    #  categorize = Categorize('f:/birdiesdata2/','f:/birdiesdata2')
    #  categorize = Categorize('/content/drive/My Drive/Colab Notebooks/BirdiesData/train',
    #           '/content/drive/My Drive/Colab Notebooks/')
    categorize = Categorize('/content/drive/My Drive/Colab Notebooks/train',
               '/content/drive/My Drive/Colab Notebooks/train')
    #  categorize.build_directories('val')
    #  categorize.create_test("test",2)
    #  Categorize('h:/birdiesdata/train', 'h:/birdiesdata')
    categorize.is_it_a_bird()
