from google_images_download import google_images_download

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


def downloadimages(query):
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


# Driver Code
i = 99
for query in search_queries:
  downloadimages(query)
  print()
#  iimage = Image.open(BytesIO(response.content))
#  #plt.imshow(iimage)
#  i + i+1
#  iimage.save(loc_data + query + str(i) + '.jpg')

