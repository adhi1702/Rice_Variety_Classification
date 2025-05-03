import tensorflow as tf

model = tf.keras.models.load_model("models/complete_model.h5", compile=False)
print("Model input shape:", model.input_shape)
