import os
from flask import (Flask, request, jsonify,
                   send_from_directory)
import utils.general as gen
from skimage import io

import image_registration
import cv2

UPLOAD_DIRECTORY = os.path.join("static", "images")

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

api = Flask(__name__)

@api.route("/files", methods=["GET"])
def list_files():
    """

    :return:
    """
    files = gen.read_listdir(UPLOAD_DIRECTORY)
    return jsonify(list(files))

@api.route("/files/<path:path>", methods=["GET"])
def get_file(path):
    """

    :param path:
    :return:
    """
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

@api.route("/files/registration", methods=["POST"])
def post_file():
    """

    :return:
    """
    if "cut1" not in request.files or "cut2" not in request.files:
        response = {
            "status": 400,
            "message": "Faltan archivos o no se enviaron"
        }
        return jsonify(response), 400
    method = int(request.args.get("method")) or 1
    cut1 = request.files["cut1"]
    cut2 = request.files["cut2"]
    cut1.save(os.path.join(UPLOAD_DIRECTORY, "cut1.png"))
    cut2.save(os.path.join(UPLOAD_DIRECTORY, "cut2.png"))

    cut1 = io.imread(os.path.join(UPLOAD_DIRECTORY, "cut1.png"))
    cut2 = io.imread(os.path.join(UPLOAD_DIRECTORY, "cut2.png"))
    # El mejor es registo_4
    cimg_array = image_registration.registration_methods[method - 1](cut1, cut2)
    io.imsave(os.path.join(UPLOAD_DIRECTORY, "registered.png"),
              cimg_array)
    return send_from_directory(UPLOAD_DIRECTORY,
                               "registered.png", as_attachment=True), 201

if __name__ == "__main__":
    api.run(debug=True, port=5000)