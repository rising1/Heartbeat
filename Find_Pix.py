from google_images_download import google_images_download
import os

class findExtraPix:

    def __init__(self, rootDir, image_shortages_list):

        self.rootDir = rootDir
        self.response = google_images_download.googleimagesdownload()

        file_list = []
        f = open(os.path.join(self.rootDir, image_shortages_list))
        for line in f:
                file_path = line.rstrip("\n").split("\t")
                file_list.append(file_path)
        #print(file_list)
        for short_item in file_list:
            print(short_item)
            self.downloadimages(short_item[0],short_item[1])

    def downloadimages(self, query, no_of_images):
        # keywords is the search query
        # format is the image file format
        # limit is the number of images to be downloaded
        # print urs is to print the image file url
        # size is the image size which can
        # be specified manually ("large, medium, icon")
        # aspect ratio denotes the height width ratio
        # of images to download. ("tall, square, wide, panoramic")
        arguments = {"keywords": query + " in flight",
                 "format": "jpg",
                 "limit": (100 - int(no_of_images)),
                 "print_urls": True,
                 "size": "medium",
                 "type": "photo",
                 "aspect_ratio": "square",
                 "delay": 0.1,
                 "output_directory": os.path.join("F:/top-up-images" )  }


        try:
                self.response.download(arguments)

            # Handling File NotFound Error
        except FileNotFoundError:
                arguments = {"keywords": query + " in flight",
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
    findExtraPix("F:/train","image_count.txt")


