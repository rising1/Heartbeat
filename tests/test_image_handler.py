import os

import constants
from bird_image_predictor import image_handler, setup

class TestImageHandler:

    # @pytest.fixture(autouse=True)
    # def setup(self):
    #     model_builder.load_and_populate_model(constants.BIRDIES_MODEL)

    def test_givenbirdpic_whenPostRequestToUploaderEndpoint_theReturnsPrediction(self):
        # given
        actual_predictions = []
        images_dir = os.path.join(constants.RESOURCES_ROOT, 'bird_pictures')

        # when
        for bird_pic in os.listdir(images_dir):
            prediction = image_handler.handle(os.path.join(images_dir, bird_pic))
            actual_predictions.append(prediction)

        # then
        assert "Duck" in actual_predictions[0]
        assert "robin" in actual_predictions[1]
        assert "swan" in actual_predictions[2]







