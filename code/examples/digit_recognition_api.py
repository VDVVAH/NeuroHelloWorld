import uvicorn

from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import File

from code.examples.server import Server

from digit_recognition import DigitRecognition

from convolutional_digit_recognition import ConvolutionalDigitRecognition


class DBClass:
    def __init__(self):
        self.digit_recognition = DigitRecognition()
        self.convolutional_digit_recognition = ConvolutionalDigitRecognition()

db = DBClass()


class WebUI(metaclass=Server):
    class Post:
        @staticmethod
        def prediction(image: bytes = File()):
            return JSONResponse({"prediction": db.digit_recognition.apply(image)}, status_code=200)

        @staticmethod
        def convolutional_prediction(image: bytes = File()):
            return JSONResponse({"prediction": db.convolutional_digit_recognition.apply(image)}, status_code=200)

    @staticmethod
    def home():
        with open("templates/digit_recognition.html") as file:
            db.digit_recognition.train()
            db.convolutional_digit_recognition.train()
            return HTMLResponse(file.read(), status_code=200)


if __name__ == "__main__":
    uvicorn.run(app=WebUI.app, host="0.0.0.0", port=8080)