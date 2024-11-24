import numpy as np

import io

from PIL import Image

from pathlib import Path

from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from keras.api.models import Sequential, load_model
from keras.api.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

from RecognitionNeuroNetwork import RecognitionNeuroNetwork


class ConvolutionalDigitRecognition(RecognitionNeuroNetwork):
    def __init__(self, number_of_training: int = 5) -> None:
        self.INPUT_SHAPE = (28, 28)                             # Размер изображения в пикселях.
        self.OPTIMIZER = "adam"                                 # Функция оптимизации при обучении.
        self.LOSS_FUNCTION = "categorical_crossentropy"  # Функция ошибки.
        self.METRICS = ["accuracy"]                             # Метрики при обучении (для логирования).
        self.SAVE_FILE = "convolutional_mnist_model.keras"      # Имя файла для сохранения модели.
        self.TRAINING_MAX_VALUE: float = 255.0                  # Максимальное значение цвета у пикселя в входных данных.
        self.BATCH_SIZE = 32                                    # Размер мини-батча для обучения

        self.number_of_training = number_of_training
        self.layers = [
            Input(shape=(*self.INPUT_SHAPE, 1)),                    # Принимает данные на входе
            Conv2D(32, (3, 3), activation='relu'),  # Применяет 32 фильтра 3 x 3 для получение признаков
            MaxPooling2D((2, 2)),                                   # Выбирает максимальные значения в областях 2 x 2 уменьшая обьём данных
            Conv2D(64, (3, 3), activation='relu'),  # Применяет 64 фильтра 3 x 3 извлекая более важные принаки
            MaxPooling2D((2, 2)),                                   # Ещё одна свёртка данных
            Flatten(),                                              # Преобразует матрицу в вектор для обработки полносвязными слоями
            Dense(64, activation='relu'),                      # Обрабатывет вектор выделяя итоговые признаки
            Dense(10, activation='relu')                       # Приводит к формату выходных данных
        ]
        #   WILL BE USED
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.x_test = None
        self.model = None

    def _prepare_data(self) -> None:
        """
        Подготавливает данные для обучения.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #  mnist - стандартный набор данных из keras для обучения нейросетей
        self.x_train = x_train.reshape((x_train.shape[0], *self.INPUT_SHAPE, 1))
        self.x_test = x_test.reshape((x_test.shape[0], *self.INPUT_SHAPE, 1))
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)

        self.x_train = self.x_train.astype('float32') / self.TRAINING_MAX_VALUE
        self.x_test = self.x_test.astype('float32') / self.TRAINING_MAX_VALUE


    def _create_model(self):
        """
        Компилирует созданную модель нейросети.
        """
        return Sequential(self.layers)

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
            self.x_train,
            self.y_train,
            epochs=self.number_of_training,
            validation_data=(self.x_test, self.y_test),
            batch_size=self.BATCH_SIZE,
            verbose=1
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
            ).reshape((1, *self.INPUT_SHAPE, 1)) / self.TRAINING_MAX_VALUE
        )

        return int(np.argmax(predictions))
