import io
import numpy as np

from keras.api.datasets import mnist
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten
from keras.api.utils import to_categorical
from keras.api.models import load_model
from PIL import Image

from Perceptron import Perceptron, Layer
# Воспользовался https://github.com/blaze-arch/MNIST-Classifier/blob/main/templates

class DigitRecognition(Perceptron):
    def __init__(self,
                 number_of_training: int = 10,
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
        return Sequential(
            Flatten(input_shape=self.INPUT_SHAPE),
            *[Dense(**layer.to_kwargs_dict()) for layer in self.layers]
        )

    def _compile_model(self, model: Sequential) -> Sequential:
        """
        Компилирует созданную модель нейросети.
        """
        model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS_FUNCTION, metrics=self.METRICS)
        return model

    def train(self):
        """
        Создаёт, компилирует, обучает и сохраняет модель нейросети.
        """
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

    def apply(self, data, **kwargs) -> int:
        """
        Применяет модель нейросети к данным.
        В случае отсутствия модели вызывает train().
        """
        if not self.model:
            self.train()

        predictions = self.model.predict(
            np.array(
                Image.open(io.BytesIO(data))
                    .resize(self.INPUT_SHAPE)
            ).reshape((1, 28, 28, 1))
        )

        return np.argmax(predictions)

