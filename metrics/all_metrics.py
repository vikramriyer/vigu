class all_metrics():

  def __init__():
    import numpy as np
    import math

  def root_mean_squared_error(actuals, predictions):
    '''
      actuals: numpy array of actual values or labels
      predictions: numpy array of predictions made by the model
    '''
    return math.sqrt(np.sum(np.square(actuals - predictions)))

  def root_mean_squared_log_error(actuals, predictions):
    '''
      actuals: numpy array of actual values or labels
      predictions: numpy array of predictions made by the model
    '''
    return math.sqrt(np.sum(np.square(np.log(actuals) - np.log(predictions))))
