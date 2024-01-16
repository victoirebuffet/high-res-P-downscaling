import matplotlib.pyplot as plt
import xarray as xr
from models import predict

def remove_spines(ax):
    ax.set_frame_on(False)
    
    
def infer(model, x_test, y_test, batch_size, key, pred_name, loss, thres):
    
    simple_cnn_prediction = predict(model, x_test, y_test, batch_size=batch_size, key=key, pred_name =pred_name, loss=loss , thres=thres)
    simple_cnn_prediction = simple_cnn_prediction.unstack()
    simple_cnn_prediction = simple_cnn_prediction.reindex(lon = sorted(simple_cnn_prediction.lon.values))
    
    return simple_cnn_prediction