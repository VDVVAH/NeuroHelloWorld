import io
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from PIL import Image
from fastapi import Form, File
from keras.api.datasets import mnist
from keras.api.layers import Dense, Flatten
from keras.api.models import Sequential
from keras.api.models import load_model
from keras.api.utils import to_categorical
from fastapi.responses import HTMLResponse, JSONResponse

from Perceptron import Perceptron, Layer
from server import Server

# Воспользовался https://github.com/blaze-arch/MNIST-Classifier/blob/main/templates

class DigitRecognition(Perceptron):
    def __init__(self,
                 number_of_training: int = 30,
                 layers: tuple[Layer] = (
                         Layer(128, "relu"),
                         Layer(64, "relu"),
                         Layer(10, "softmax")
                    )
                 ) -> None:
        self.number_of_training: int = number_of_training   # Количество эпох обучения
        self.layers: tuple[Layer] = layers                  # Слои нейросети
        #  CONSTANTS
        self.TRAINING_MAX_VALUE: float = 255.0              # Максимальное значение цвета у пикселя в входных данных.
        self.INPUT_SHAPE = (28, 28)                         # Размер изображения в пикселях.
        self.OPTIMIZER = "adam"                             # Функция оптимизации при обучении.
        self.LOSS_FUNCTION = "categorical_crossentropy"     # Функция ошибки.
        self.METRICS = ["accuracy"]                         # Метрики при обучении (для логирования).
        self.SAVE_FILE = "mnist_model.keras"                # Имя файла для сохранения модели.
        #   WILL BE USED
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.train_labels = None
        self.model = None

    def _prepare_data(self) -> None:
        """
        Подготавливает данные для обучения.
        """
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        #  mnist - стандартный набор данных из keras для обучения нейросетей
        self.train_images = train_images / self.TRAINING_MAX_VALUE
        self.test_images = test_images / self.TRAINING_MAX_VALUE
        self.train_labels = to_categorical(train_labels)
        self.test_labels = to_categorical(test_labels)

    def _create_model(self) -> Sequential:
        """
        Создаёт модель нейросети.
        """
        return Sequential([
            Flatten(input_shape=self.INPUT_SHAPE),
            *[Dense(**layer.to_kwargs_dict()) for layer in self.layers]
            ])

    def _compile_model(self, model: Sequential) -> Sequential:
        """
        Компилирует созданную модель нейросети.
        """
        model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS_FUNCTION, metrics=self.METRICS)
        return model

    def train(self) -> None:
        """
        Создаёт, компилирует, обучает и сохраняет модель нейросети.
        """
        if Path(__file__).with_name(self.SAVE_FILE).exists():
            self.model = load_model(filepath=self.SAVE_FILE)
            return

        self._prepare_data()
        self.model = self._compile_model(
            self._create_model()
        )
        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs=self.number_of_training,
            validation_data=(self.test_images, self.test_labels)
        )
        self.model.save(self.SAVE_FILE)

    def apply(self, data: bytes, **kwargs) -> int:
        """
        Применяет модель нейросети к данным.
        В случае отсутствия модели вызывает train().
        """
        if not self.model:
            self.train()

        predictions = self.model.predict(
            np.array(
                Image.open(io.BytesIO(data))
                    .convert("L")
                    .resize(self.INPUT_SHAPE)
            ).reshape((1, 28, 28, 1)) / 255.0
        )

        return int(np.argmax(predictions))


class DBInClass:
    def __init__(self):
        self.model = DigitRecognition()

db = DBInClass()

class WebUI(metaclass=Server):
    class Post:
        @staticmethod
        def prediction(image: bytes = File()):
            return JSONResponse({"prediction": db.model.apply(image)}, status_code=200)

    @staticmethod
    def home():
        with open("templates/digit_recognition.html") as file:
            db.model.train()
            return HTMLResponse(file.read(), status_code=200)


if __name__ == "__main__":
    uvicorn.run(app=WebUI.app, host="0.0.0.0", port=8080)