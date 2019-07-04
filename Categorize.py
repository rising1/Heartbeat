import os

class Categorize:
    def __init__(self, rootDir, catType):
        self.rootDir = rootDir
        self.catType = catType
        for dirPath, dirNames, fileNames in os.walk(self.rootDir):