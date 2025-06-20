import numpy as np
from scipy.signal.windows import exponential

def create_win(M, center=0, tau=3):
    win = exponential(M, center, tau, False)
    print('++ Create Window: Window Values [%s]' % str(win))
    return win[:, np.newaxis]

class CircularBuffer:
    """Used for managing windowing in Matcher"""
    def __init__(self, Nv, size):
        self.insert_idx = 0
        self.size = size
        self.buffer = np.zeros((Nv, size))
        self.full = False
    
    def update(self, data):
        self.buffer[:, self.insert_idx] = data.squeeze()
        self.insert_idx = (self.insert_idx + 1) % self.size

        if self.insert_idx == 0 and not self.full:
            self.full = True

        if self.full:
            return np.concatenate([self.buffer[:, self.insert_idx:], self.buffer[:, :self.insert_idx]], axis=1)
        
        else:
            return None

def rt_svrscore_vol(data, SVRs, caps_labels):
    """
    Compute SVR scores using pretrained models.

    Parameters
    ----------
    data : np.ndarray
        The input data to be used for making predictions.
    SVRs : dict
        A dictionary of trained Support Vector Regressor (SVR) models, where the keys are 
        label names and the values are the corresponding SVR models.
    caps_labels : list of str
        A list of labels corresponding to the SVRs in `SVRs`. The function will use these 
        labels to predict the values from the respective SVRs.

    Returns
    -------

    np.ndarray
        The predicted values from each SVR for each label.
    """
    out = []

    for cap_lab in caps_labels:
        out.append(SVRs[cap_lab].predict(data[:,np.newaxis].T)[0])

    return np.array(out)[:,np.newaxis]

def rt_maskscore_vol(data, inputs, labels):
    out = []
    masked_templates = inputs["masked_templates"].item()
    masks = inputs["masks"].item()
    voxel_counts = inputs["voxel_counts"].item()

    for name in labels:
        mask = masks[name]
        template = masked_templates[name]
        masked_data = data[mask]        
        out.append(np.dot(template, masked_data) / voxel_counts[name])
    return np.array(out)[:, np.newaxis]
