import os
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime

def get_param_from_layer(model, layer_name):
    layer = model.get_layer(name=layer_name)
    weights = layer.get_weights()
    # Concatenate the weights into a single numpy array
    params = np.concatenate([w.flatten() for w in weights])
    return params

def export_proof(model, filename='proof.pkl'):
    param_1 = get_param_from_layer(model, 'check_layer_1').reshape((1, -1))
    param_2 = get_param_from_layer(model, 'check_layer_2').reshape((1, -1))
    proof = np.concatenate([param_1.T, param_2.T], axis=0)
    with open(filename, 'wb') as f:
        pickle.dump(proof, f, pickle.HIGHEST_PROTOCOL)
    print(f"Proof exported to {filename}")

def train_model(model, train_x, train_y, epochs=50, batch_size=32, log_dir="logs"):
    # set up the tensorboard callback for reviewing real-time progress
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback])
    return model
