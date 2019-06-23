from google_images_download import google_images_download
import os


class findPix:
    global search_query
    global response
    global skipDirectories
    loc_data = 'C:/Users/phfro/Documents/python/data/'
    response = google_images_download.googleimagesdownload()

    def __init__(self, keywords_file_path,skipDirectories):
        # creating object
        self.keywords_file_path = keywords_file_path
        self.skipDirectories = skipDirectories
        response = google_images_download.googleimagesdownload()
        if  (os.path.exists(self.loc_data + self.keywords_file_path)):
            with open(self.loc_data + self.keywords_file_path, "r") as f:
                self.search_query = f.read().splitlines()
            for query in self.search_query:
                #  print("query=",query)
                self.downloadimages(query)

        else:
            print("keyword file not present")



    def downloadimages(self, query):
        # keywords is the search query
        # format is the image file format
        # limit is the number of images to be downloaded
        # print urs is to print the image file url
        # size is the image size which can
        # be specified manually ("large, medium, icon")
        # aspect ratio denotes the height width ratio
        # of images to download. ("tall, square, wide, panoramic")
        arguments = {"keywords": query,
                 "format": "jpg",
                 "limit": 99,
                 "print_urls": True,
                 "size": "medium",
                 "type": "photo",
                 "aspect_ratio": "square",
                 "output_directory": self.loc_data + "BirdiesData/train/" }
        dataPath = self.loc_data + "BirdiesData/train/" + query
        print(os.path.exists(dataPath))
        if not(os.path.exists(dataPath)and self.skipDirectories):
            try:
                response.download(arguments)

            # Handling File NotFound Error
            except FileNotFoundError:
                arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": 4,
                     "print_urls": True,
                     "size": "medium",
                     "output_directory": self.loc_data + self.downloadimages(query)}

            # Providing arguments for the searched query
            try:
                # Downloading the photos based
                # on the given arguments
                self.response.download(arguments)
            except:
                pass

# train(num_epochs)
if __name__ == "__main__":
    findPix("british_birds.txt",True)
#  iimage = Image.open(BytesIO(response.content))
#  #plt.imshow(iimage)
#  i + i+1
#  iimage.save(loc_data + query + str(i) + '.jpg')

# clean directories
#import os
#rootDir = '/content/drive/My Drive/' \
#  'Colab Notebooks/BirdiesData/birdpix/'
#for dirName, subdirList, fileList in os.walk(rootDir):
#    #print('Found directory: %s' % dirName)
#    for filename in fileList:
#      #print('\t%s' % filename)
#      testFile = dirName + '/' + filename
#      print(testFile)
#      try:
#        im = Image.open(testFile)
#        im.verify() #I perform also verify, don't know if he sees other types o defects
#        im.close() #reload is necessary in my case
#        #im = Image.load(filename)
#        #im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#        #im.close()
#      except:
#        os.remove(testFile)