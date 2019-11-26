from google_images_download import google_images_download
import os
from PIL import Image
import BingPix

class findExtraPix:

    def __init__(self, rootDir, image_shortages_list, keyword_mod, load_new_pix):

        self.rootDir = rootDir
        # self.response = google_images_download.googleimagesdownload()


        file_list = []
        f = open(os.path.join(self.rootDir, image_shortages_list))
        for line in f:
                file_path = line.rstrip("\n").split("\t")
                file_list.append(file_path)
        #print(file_list)
        i = 0
        for short_item in file_list:
            print(short_item)
            print(short_item[0] + keyword_mod)
            # self.downloadimages(short_item[0],short_item[1])
            if (load_new_pix):
                BingPix.pre_prep(self.rootDir + "/" + short_item[0], True, (short_item[0] + keyword_mod), short_item[1],20)
            else:
                for sfileNames in next(os.walk(self.rootDir + '/' + short_item[0]))[2]:
                    print("working with ", self.rootDir + '/' + short_item[0])
                    while i < (100 - int(short_item[1])):
                            'code to copy the existing pictures and rotate them'
                            file_path = self.rootDir + '/' + short_item[0] + '/' + sfileNames
                            print(file_path, "i= ",i)
                            if os.path.isfile(file_path):
                                try:
                                    im = Image.open(file_path)
                                    print("image successfully opened")
                                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                                    print("image successfully transposed")
                                    # im.save( file_path + "flipped.jpg")
                                    print("passed flipped_", file_path)
                                    i = i + 1
                                except Exception as e:
                                    print(str(e))






    def downloadimages(self, query, no_of_images,keyword_mod):
        # keywords is the search query
        # format is the image file format
        # limit is the number of images to be downloaded
        # print urs is to print the image file url
        # size is the image size which can
        # be specified manually ("large, medium, icon")
        # aspect ratio denotes the height width ratio
        # of images to download. ("tall, square, wide, panoramic")
        arguments = {"keywords": (query + keyword_mod),
                 "format": "jpg",
                 "limit": (100 - int(no_of_images)),
                 "print_urls": True,
                 "size": "medium",
                 "type": "photo",
                 "aspect_ratio": "square",
                 "delay": 0.1,
                 "output_directory": os.path.join("F:/top-up-images" )  }

        if (int(no_of_images)) < 100:
            try:

                self.response.download(arguments)

                # Handling File NotFound Error
            except FileNotFoundError:
                arguments = {"keywords": query + " nesting",
                     "format": "jpg",
                     "limit": (100 - int(no_of_images)),
                     "print_urls": True,
                     "size": "medium",
                     "output_directory": os.path.join("F:/top-up-images" )}

            # Providing arguments for the searched query
            try:
                # Downloading the photos based
                # on the given arguments
                self.response.download(arguments)
            except:
                pass


if __name__ == "__main__":
    # findExtraPix("F:/train_2","image_count.txt"," bird", False)
    findExtraPix("C:/Users/phfro/PycharmProjects/Heartbeat/train","image_count.txt"," bird", False)
    # findExtraPix("F:/train","image_count.txt"," bird", False)
    # findExtraPix("E:/top-up-images","image_count.txt", " bird")
    # findExtraPix("D:/top-up-images","image_count.txt"," bird")
