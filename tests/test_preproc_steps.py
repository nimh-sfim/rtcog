import numpy as np
from rtcog.preproc.preproc_steps import EMAStep

class DummyPipeline:
    def __init__(self, data, n):
        self.Data_FromAFNI = data
        self.t = data.shape[1] - 1
        self.n = n

def test_EMAStep_first():
    data = np.array([[1, 2, 3],
                     [4, 5, 6]])

    pipeline = DummyPipeline(data, n=1)
    ema = EMAStep(save=False, alpha=0.98)

    out = ema._run(pipeline)

    assert out.shape == (2, 1)
    assert ema.filt.shape == (2, 1)

def test_EMAStep_second():
    data = np.array([[1, 2, 3],
                     [4, 5, 6]])

    ema = EMAStep(save=False, alpha=0.98)

    pipeline1 = DummyPipeline(data, n=1)
    ema._run(pipeline1)

    pipeline2 = DummyPipeline(data, n=2)
    out = ema._run(pipeline2)

    assert out.shape == (2, 1)
    assert ema.filt.shape == (2, 1)