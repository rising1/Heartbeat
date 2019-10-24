from google_images_download import google_images_download
import os


class findPix:

    loc_data = 'C:/Users/phfro/PycharmProjects/Heartbeat'


    def __init__(self, keywords_file_path,subdir,skipDirectories):
        # creating object
        self.keywords_file_path = keywords_file_path
        self.skipDirectories = skipDirectories
        self.subdir = subdir
        self.response = google_images_download.googleimagesdownload()
        if  (os.path.exists(self.loc_data + "/" + self.keywords_file_path)):
            with open(self.loc_data + "/" + self.keywords_file_path, "r") as f:
                self.search_query = f.read().splitlines()
                self.search_query.sort()
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
                 "limit": 2,
                 "print_urls": True,
                 "size": "medium",
                 "type": "photo",
                 "aspect_ratio": "square",
                 "delay": 0.1,
                 "output_directory": self.loc_data + self.subdir  }
        dataPath = self.loc_data + self.subdir + query
        print("dataPath=",dataPath)
        print(os.path.exists(dataPath))
        if not(os.path.exists(dataPath)and self.skipDirectories):
            try:
                self.response.download(arguments)

            # Handling File NotFound Error
            except FileNotFoundError:
                arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": 1,
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
    findPix("bird_dir_list.txt","train",True)
#  iimage = Image.open(BytesIO(response.content))
#  #plt.imshow(iimage)
#  i + i+1
#  iimage.save(loc_data + query + str(i) + '.jpg')

