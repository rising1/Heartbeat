from google_images_download import google_images_download
import ImageType
import os
from shutil import copyfile

loc_data = '/content/drive/My Drive/Colab Notebooks/trial'
# creating object
response = google_images_download.googleimagesdownload()

search_queries = \
  [

    'sparrowhawk in flight',
    'kestrel in flight',
    'common buzzard in flight',
    'golden eagle in flight',
    'red kite in flight',
    'peregrine falcon'

  ]

class Load_pix():
    image_type = ImageType.ImageType()
    file_list = []
    f = open("Bird_Index.txt")
    for line in f:
        file_path = line.rstrip("\n").split(",")
        #  file_path = line.split(",")
        file_list.append(file_path)
        print(file_list)
    def __init__(self,search_queries,loc_data,src_data):
        self.search_queries = search_queries
        self.loc_data = loc_data
        self.src_data = src_data
        # Driver Code
        for query in self.search_queries:
            if int(query[1]) < 100:
                no_of_shorts = 100 - int(query[1])
                self.downloadimages(query[0],no_of_shorts)
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
        arguments = {"keywords": query  + "in flight",
               "format": "jpg",
               "limit": no_of_shorts,
               "print_urls": True,
               "size": ">400*300",
               "type": "photo",
               "aspect_ratio": "square",
               "output_directory": loc_data}
        try:
            response.download(arguments)

         # Handling File NotFound Error
        except FileNotFoundError:
            arguments = {"keywords": query,
                 "format": "jpg",
                 "limit": 4,
                 "print_urls": True,
                 "size": ">400*300",
                 "output_directory": loc_data}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass


        file_path = os.path.join(loc_data, query)
        if os.path.isfile(file_path):
            try:
                result = self.image_type.predict_image(file_path)
                # look up the result in an index and check
                #whether a bird or not
                if result in self.file_list:
                    copyfile(file_path,self.src_data + query +
                             os.path.basename(file_path) )
            except RuntimeError:
                result = "PREDICT_FAILURE"


