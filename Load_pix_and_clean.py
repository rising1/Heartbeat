from google_images_download import google_images_download


loc_data = '/content/drive/My Drive/Colab Notebooks/BirdiesData/birdpix/raw_downloads/'
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

import json
import os
import time
import requests
from PIL import Image
from io import StringIO
from requests.exceptions import ConnectionError

def go(query, path):
  """Download full size images from Google image search.
  Don't print or republish images without permission.
  I used this to train a learning algorithm.
  """
  BASE_URL = 'https://ajax.googleapis.com/ajax/services/search/images?'\
             'v=1.0&q=' + query + '&start=%d'

  BASE_PATH = os.path.join(path, query)
  print(BASE_PATH)
  if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

  start = 0 # Google's start query string parameter for pagination.
  while start < 60: # Google will only return a max of 56 results.
    r = requests.get(BASE_URL % start)
    for image_info in json.loads(r.text)['responseData']['results']:
      url = image_info['unescapedUrl']
      try:
        image_r = requests.get(url)
      except ConnectionError:
        print ('could not download %s' % url)
        continue

      # Remove file-system path characters from name.
      title = image_info['titleNoFormatting'].replace('/', '').replace('\\', '')

      file = open(os.path.join(BASE_PATH, '%s.jpg') % title, 'w')
      try:
        Image.open(StringIO(image_r.content)).save(file, 'JPEG')
      except IOError:
        # Throw away some gifs...blegh.
        print ('could not save %s' % url)
        continue
      finally:
        file.close()

    print (start)
    start += 4 # 4 images per page.

    # Be nice to Google and they'll be nice back :)
    time.sleep(1.5)

# Example use
go('landscape', '/content/drive/My Drive/Colab Notebooks/BirdiesData/birdpix/raw_downloads')