import tensorflowjs as tfjs
from tensorflow.keras.models import load_model
import sys

_, model_path = sys.argv

model = load_model(model_path)

tfjs.converters.save_keras_model(model, "jsmodels")