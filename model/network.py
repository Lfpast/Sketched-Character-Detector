from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Dropout

def create_model(input_shape=(224, 224, 3)):
  """Create a CNN model for predicting the bounding box for drawings in an image
  input_shape: the targeted image size

  return the model architecture
  """
  # load the pre-trained ResNet50 model without the top classification layer
  base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
  # freeze the base model layers
  base_model.trainable = False
  # add custom layers on top for bounding box prediction
  model_x = base_model.output

  # Add convolutional layers
  model_x = Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding="same")(model_x)
  model_x = Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same")(model_x)
  model_x = Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same")(model_x)

  model_x = GlobalAveragePooling2D()(model_x)  # use global average pooling to flatten the output
  model_x = Dropout(0.5)(model_x) # randomly drop out weights to avoid overfitting
  model_x = Dense(64, activation='relu', kernel_initializer='random_normal', name="check_layer_1")(model_x)  # add a fully connected layer
  model_x = Dense(32, activation='relu', kernel_initializer='random_normal', name="check_layer_2")(model_x)  # add a fully connected layer

  # Add a fully connected layer for predicting (x, y, w, h) for bounding box
  model_x = Dense(4, activation="sigmoid", kernel_initializer='random_normal', name="output_layer")(model_x)

  model = Model(inputs=base_model.input, outputs=model_x)
  return model
