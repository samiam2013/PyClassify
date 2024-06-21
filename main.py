# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
import json
import base64
import time
import os
import numpy as np

load_dotenv()
print("Loading tensorflow...")
# TF freaks about lack of cuda gpu without this
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
print("Tensorflow loaded.")

print("Loading ResNet50 model...")
# Load the ResNet50 model pre-trained on ImageNet data globally
model = ResNet50(weights='imagenet')
# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print("ResNet50 model loaded.")

hostName = os.getenv("HOSTNAME") or "localhost"
serverPort = int(os.getenv("PORT")) or 8080
imagesPath = os.getenv("IMAGES_PATH") or ""

if(not imagesPath == "" and not os.path.exists("imagesPath")):
    os.makedirs(imagesPath)

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            # get ./submit.py and send it back  
            with open("./submit.html", "r") as f:
                self.wfile.write(bytes(f.read(), "utf-8"))
        except Exception as e:
            error(self, 500, str(e))

    def do_POST(self):
        try:
            # first confirm that the request is for /classify 
            if self.path != "/classify": 
                error_json(self, 400, "Bad request, must POST to /classify")
            content_length = int(self.headers['Content-Length'])
            data_input = self.rfile.read(content_length)
            # decode from json to get image
            data_input = json.loads(data_input.decode("utf-8"))
            image = data_input.get("image")
            if image is None:
                error_json(self, 400, "Bad request, must include 'image' (b64 jpeg)")

            # strip the MIME type and base64 designation from the multipart string then try to save
            # "data:image/jpeg;base64,/9j/4AAQS....
            if image.startswith("data:image/jpeg;base64,"):
                image = image[len("data:image/jpeg;base64,"):]
            
            # get the current time + .jpg save the image
            filename = imagesPath + "/" + str(time.time()) + ".jpg"
            with open(filename, "wb") as fh:
                fh.write(base64.b64decode(image))

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            classes = get_image_classes(filename)
            # delete the file
            os.remove(filename) if imagesPath == "" else None
            if classes is None:
                fh.close()
                error_json(self, 500, "Internal server error")
            response = {
                "success": True,
                "upload_size": len(data_input),
                "classifications": classes
            }
            json_response = json.dumps(response)
            self.wfile.write(bytes(json_response, "utf-8"))
            open(filename + ".json", "wb").write(bytes(json_response, "utf-8")) if imagesPath == "" else None
        except Exception as e:
            error_json(self, 500, str(e))

def get_image_classes(img_path):
    # this is where we need to take that file handle and run it through our model
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array,verbose=0)

    preds = decode_predictions(prediction, top=10)[0]

    label_probs = dict()
    for pred in preds:
        label_probs[pred[1]] = str(pred[2])
    return label_probs


def error(server, code, msg):
    server.send_response(code)
    server.send_header("Content-type", "text/html")
    server.end_headers()
    server.wfile.write(bytes("<html><head><title>classify server</title></head>", "utf-8"))
    server.wfile.write(bytes("<body><p>%s</p></body></html>" % msg, "utf-8"))

def error_json(server, code, msg):
    server.send_response(code)
    server.send_header("Content-type", "application/json")
    server.end_headers()
    response = {
        "success": False,
        "error": msg
    }
    json_response = json.dumps(response)
    server.wfile.write(bytes(json_response, "utf-8"))

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")