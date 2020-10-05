import os

import pytest

import constants
from bird_image_predictor import image_handler
from model import model_builder


class TestImageHandler:

    @pytest.fixture(autouse=True)
    def setup(self):
        model_builder.load_and_populate_model(constants.BIRDIES_MODEL)

    def test_givenbirdpic_whenPostRequestToUploaderEndpoint_theReturnsPrediction(self):
        #given
        image_filepath = os.path.join(constants.RESOURCES_ROOT, 'Mallard.jpg')

        #when
        prediction = image_handler.handle(image_filepath)

        #then
        assert "Duck" in prediction






