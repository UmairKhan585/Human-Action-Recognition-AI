from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense

from PIL import Image
import numpy as np

# Define the Flask app
app = Flask(__name__)
label_map = {
    0: "sitting",
    1: "using laptop",
    2: "hugging",
    3: "sleeping",
    4: "drinking",
    5: "clapping",
    6: "dancing",
    7: "cycling",
    8: "calling",
    9: "laughing",
    10: "eating",
    11: "fighting",
    12: "listening_to_music",
    13: "running",
    14: "texting"
}

# Load the trained model
cnn_model = tf.keras.models.Sequential()
# cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(256,activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(15, activation='softmax'))


# Load the weights into the model
cnn_model.load_weights('13-6-2023 Model\colab_cnn_weights_model_4.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the image processing route
@app.route('/process', methods=['POST'])
def process_image():
    # Get the uploaded image file from the request
    image_file = request.files['image']

    # Open and preprocess the image
    img = Image.open(image_file)
    img = img.resize((160, 160))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Perform action recognition
    predictions = cnn_model.predict(img_array)
    action_label = np.argmax(predictions)

    # Return the action label as the result
    return render_template('index.html', action_label=label_map[action_label])

# Run the Flask app
if __name__ == '__main__':
    app.run()




























# from flask import Flask, request, render_template
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# # Define the Flask app
# app = Flask(__name__)
# label_map = {
#     0: "sitting",
#     1: "using laptop",
#     2: "hugging",
#     3: "sleeping",
#     4: "drinking",
#     5: "clapping",
#     6: "dancing",
#     7: "cycling",
#     8: "calling",
#     9: "laughing",
#     10: "eating",
#     11: "fighting",
#     12: "listening_to_music",
#     13: "running",
#     14: "texting"
# }
# # Load the trained model
# model = tf.keras.models.Sequential()
# model.add(tf.keras.applications.VGG16(include_top=False,
#                    input_shape=(160,160,3),
#                    pooling='avg',classes=15,
#                    weights='imagenet'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.Dense(15, activation='softmax'))

# # Load the weights into the model
# model.load_weights('colab_cnn_weights_model_3.h5')

# # Define the home route
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Define the image processing route
# @app.route('/process', methods=['POST'])
# def process_image():
#     # Get the uploaded image file from the request
#     image_file = request.files['image']

#     # Open and preprocess the image
#     img = Image.open(image_file)
#     img = img.resize((160, 160))
#     img = img.convert('RGB')
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Perform action recognition
#     predictions = model.predict(img_array)
#     print("Predictions",predictions)
#     action_label = np.argmax(predictions)
#     print("dvnbhjbujhyb")
#     # Return the action label as the result
#     return render_template('index.html', action_label=label_map[action_label])

# # Run the Flask app
# if __name__ == '__main__':
#     app.run()
