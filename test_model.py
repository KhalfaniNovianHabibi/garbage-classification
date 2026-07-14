import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print(tf.__version__)

model = tf.keras.models.load_model(
    "garbage_classification_mobilenetv2.keras",
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("SUCCESS")