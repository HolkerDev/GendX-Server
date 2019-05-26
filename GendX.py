import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=cpu, optimizer=fast_compile'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model
from keras.preprocessing import image


class GendXModel():
    def __init__(self, path):
        self.model = load_model(path)

    def predict(self, path):
        pre_image = self.__convert_image(path)
        result = self.model.predict(pre_image)
        if result[0] == 0:
            return 'man'
        elif result[0] == 1:
            return 'woman'
        else:
            return 'unknown'

    def __convert_image(self, path):
        files = image.load_img(path, target_size=(64, 64))

        # converting to array
        test_image = image.img_to_array(files)

        # extend by 1 dimension to (64,64,3)
        converted_image = np.expand_dims(test_image, axis=0)
        return converted_image
