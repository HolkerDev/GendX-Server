import base64
import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=cpu, floatX=float32, optimizer=fast_compile'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from flask import Flask, render_template, request, Response
from flask_restful import Api
import jsonpickle
import CropImage
import GendX
from PIL import Image
from keras import backend as K

CASCADE_PATH = 'haarcascade_frontalface_default.xml'
GENDX_PATH = 'models/weights.06-0.35.hdf5'

PASS = '1111'

# init Flask app
app = Flask(__name__)
api = Api(app)

# global counter for saving images
counter = 0

# init GendX and OpenCV
gendX = GendX.GendXModel(GENDX_PATH)
open_cv = CropImage.OpenCV(CASCADE_PATH)


# route for documentation
@app.route('/')
def welcome():
    return render_template('documentation.html')


# route for documentation section
@app.route('/doc/with_open_cv')
def doc_open_cv():
    return render_template('opencv.html')


# route for documentation section
@app.route('/doc/clear_gendx')
def doc_without_open_cv():
    return render_template("gendx.html")


# route for recognition without opencv for 1 person
@app.route('/api/recognize')
def recognize_one_obj():
    r = request.args.get('image')

    # creating right path for file
    global counter
    image_path = 'received_one/' + str(counter) + '.jpg'

    # decode image and save it
    image_64_decode = base64.decodebytes(str(r).encode('ascii'))
    image_result = open(image_path, 'wb')
    image_result.write(image_64_decode)

    img = Image.open(image_path)

    # preparing response
    response = {
        "accuracy": "85%",
        "format": "{}".format(img.size),
        "gender": gendX.predict(image_path)
    }

    # increase counter for next file path
    counter += 1

    # convert dict to json readable response
    response_json = jsonpickle.encode(response)

    return Response(response=response_json, status=200, mimetype="application/json")


# route for recognition with open cv for anyone
@app.route('/api/recognize_with_opencv')
def recognize_with_open_cv():
    r = request.args.get('image')

    # creating right path for file
    global counter
    image_path = 'received_many/' + str(counter) + '.jpg'

    # decode image and save it
    image_64_decode = base64.decodebytes(str(r).encode('ascii'))
    image_result = open(image_path, 'wb')
    image_result.write(image_64_decode)

    # open saved image
    img = Image.open(image_path)

    # crop images and return paths
    faces_arr = open_cv.crop(image_path)

    people_json = []

    for i in range(0, len(faces_arr)):
        people_json.append({"gender": gendX.predict(faces_arr[i])})

    # json response
    response = {"accuracy": "85%",
                "format": "{}".format(img.size),
                "people": people_json}
    # increase counter for next file path
    counter += 1

    # convert dict to json readable response
    response_json = jsonpickle.encode(response)

    return Response(response=response_json, status=200, mimetype="application/json")


# route for testing
@app.route('/test')
def test():
    r = request.args.get('image')

    image_64_decode = base64.decodebytes(str(r).encode('ascii'))
    image_result = open('deer_decode.png', 'wb')
    image_result.write(image_64_decode)

    response = {
        "msg": "received {}".format(r)
    }
    response_json = jsonpickle.encode(response)

    return Response(response=response_json, status=200, mimetype="application/json")


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


# shutdown remotely
@app.route('/shutdown')
def shutdown():
    r = request.args.get("pass")
    if r == PASS:
        shutdown_server()
        return 'Server shutting down...'
    else:
        return 'No way'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
