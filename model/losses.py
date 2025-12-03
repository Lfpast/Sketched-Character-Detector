import tensorflow as tf
import keras

def se_func(x1_pred, y1_pred, w_pred, h_pred, x1_gt, y1_gt, w_gt, h_gt):
  """Compute the square error (SE) based on the tensor of (x, y, w, h) for the predicted bbox and ground truth

  return an (N * 1) tensor as the individual loss value
  """
  tlx_diff = tf.square(x1_pred - x1_gt)
  tly_diff = tf.square(y1_pred - y1_gt)
  brx_diff = tf.square(x1_pred + w_pred - x1_gt - w_gt)
  bry_diff = tf.square(y1_pred + h_pred - y1_gt - h_gt)
  SE = tlx_diff + tly_diff + brx_diff + bry_diff
  return SE

def giou_func(x1_pred, y1_pred, w_pred, h_pred, x1_gt, y1_gt, w_gt, h_gt):
  """Compute the giou loss based on the tensor of (x, y, w, h) for the predicted bbox and ground truth

  return an (N * 1) tensor as the loss value
  """
  epsilon = 1e-7
  zero_tensor = tf.zeros_like(x1_pred)
  # Calculate intersection coordinates
  I_x = tf.maximum((tf.minimum(x1_pred + w_pred, x1_gt + w_gt) - tf.maximum(x1_pred, x1_gt)),zero_tensor)
  I_y = tf.maximum((tf.minimum(y1_pred + h_pred, y1_gt + h_gt) - tf.maximum(y1_pred, y1_gt)),zero_tensor)
  I = I_x * I_y

  # Calculate union area
  U = w_pred * h_pred + w_gt * h_gt - I

  # Calculate enclosing box coordinates
  C_x = tf.maximum(x1_pred + w_pred, x1_gt + w_gt) - tf.minimum(x1_pred, x1_gt)
  C_y = tf.maximum(y1_pred + h_pred, y1_gt + h_gt) - tf.minimum(y1_pred, y1_gt)
  C = C_x * C_y

  # Calculate IoU and GIoU
  IoU = I / (U + epsilon)
  GIoU = IoU - (C - U) / (C + epsilon)

  # Calculate GIoU loss
  GIoU_loss = 1 - GIoU
  return GIoU_loss

@keras.saving.register_keras_serializable(name="loss")
def loss_func(pred, gt):
  """The loss function for model training.
  pred: a (N, 4) numpy array of predicted value for x, y, w, h
  gt: a (N, 4) numpy array of the ground truth value x, y, w, h

  return a (N, 1) numpy array of the loss value
  """
  gt = tf.convert_to_tensor(gt, dtype=tf.float32)
  pred = tf.convert_to_tensor(pred, dtype=tf.float32)

  x1_gt, y1_gt, w_gt, h_gt = tf.split(gt, 4, axis=-1)
  x1_pred, y1_pred, w_pred, h_pred = tf.split(pred, 4, axis=-1)

  # you can also try using the mse error
  #loss = se_func(x1_pred, y1_pred, w_pred, h_pred, x1_gt, y1_gt, w_gt, h_gt)
  loss = giou_func(x1_pred, y1_pred, w_pred, h_pred, x1_gt, y1_gt, w_gt, h_gt)
  return tf.reduce_mean(loss)
