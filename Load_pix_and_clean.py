from google_images_download import google_images_download
import ImageType
import os
from shutil import copyfile

''' The purpose of this file is to create the directory structure and to load pictures into the directory tree -----'''
''' It starts from a file containing keywords ----------------------------------------------------------------------'''

# loc_data = '/content/drive/My Drive/Colab Notebooks/trial'
loc_data = 'C:/Users/phfro/PycharmProjects/Heartbeat'

# creating object
response = google_images_download.googleimagesdownload()

class Load_pix():
    image_type = ImageType.ImageType()
    file_list = []
    f = open("Bird_Index.txt") # for the bird pic detector
    for line in f:
        file_path = line.rstrip("\n").split(",")
        #  file_path = line.split(",")
        file_list.append(file_path)
        print(file_list)

    def __init__(self,search_queries,loc_data,src_data):
        self.search_queries = search_queries
        self.loc_data = loc_data
        self.src_data = src_data

        if  (os.path.exists(self.loc_data + "/" + self.search_queries)):
            with open(self.loc_data + "/" + self.search_queries, "r") as f:
                self.search_query = f.read().splitlines()
                self.search_query.sort()


        # Driver Code
        for query in self.search_query:
            #if int(query[1]) < 100:
                no_of_shorts = 99
                self.downloadimages(query,no_of_shorts)
                self.remove_non_birds(query)
        #  iimage = Image.open(BytesIO(response.content))
        #  #plt.imshow(iimage)
        #  i + i+1
        #  iimage.save(loc_data + query + str(i) + '.jpg')

    def downloadimages(self,query,no_of_shorts):
        # keywords is the search query
        # format is the image file format
        # limit is the number of images to be downloaded
        # print urs is to print the image file url
        # size is the image size which can
        # be specified manually ("large, medium, icon")
        # aspect ratio denotes the height width ratio
        # of images to download. ("tall, square, wide, panoramic")
        arguments = {"keywords": query , #   + " in flight",
               "format": "jpg",
               "limit": no_of_shorts,
               "print_urls": True,
               "size": ">400*300",
               "type": "photo",
               "delay": 0.1,
               "aspect_ratio": "square",
               "output_directory": loc_data + "/train"}
        try:
            response.download(arguments)

         # Handling File NotFound Error
        except FileNotFoundError:
            arguments = {"keywords": query,
                 "format": "jpg",
                 "limit": 0,
                 "print_urls": True,
                 "size": ">400*300",
                 "output_directory": loc_data + "/train"}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass

    ''' This bit doesn't work for some reason ------------------------------------------------------------------'''
    def remove_non_birds(self,query):
        file_path = loc_data + "/train/" + query
        for sfileNames in next(os.walk(file_path))[2]:
                print("sfileNames=",sfileNames)
                if os.path.isfile(file_path):
                    try:
                        result = self.image_type.predict_image(file_path)
                        # look up the result in an index and check
                        #whether a bird or not
                        if result in self.file_list:
                            #copyfile(file_path,self.src_data + query +
                            #         os.path.basename(file_path) )
                            print("checked",os.path.basename(file_path))
                        else:
                            os.remove(file_path)

                    except RuntimeError:
                        result = "PREDICT_FAILURE"
                else:
                    print("breaking..")
                    break



if __name__ == "__main__":
    loc_data = 'C:/Users/phfro/PycharmProjects/Heartbeat'
    search_queries = "bird_dir_list.txt"
    src_data = "unused"
    pix_loader = Load_pix(search_queries,loc_data,src_data)
