import numpy as np

from rtcog.preproc.preproc_steps import EMAStep
from rtcog.preproc.preproc_steps import SnormStep

class DummyPipeline:
    def __init__(self, data=None, processed_tr=None, n=None, t=None):
        self.Data_FromAFNI = data
        self.processed_tr = processed_tr
        self.n = n

        if t is not None:
            self.t = t
        elif data is not None:
            self.t = data.shape[1] - 1
        elif processed_tr is not None:
            self.t = processed_tr.shape[1] - 1
        else:
            self.t = None


def test_EMAStep_first():
    data = np.array([[1, 2, 3],
                     [4, 5, 6]])

    pipeline = DummyPipeline(data=data, n=1)
    ema = EMAStep(save=False, alpha=0.98)

    out = ema._run(pipeline)

    assert out.shape == (2, 1)
    assert ema.filt.shape == (2, 1)


def test_EMAStep_second():
    data = np.array([[1, 2, 3, 4],
                     [4, 5, 6, 7]])

    ema = EMAStep(save=False, alpha=0.98)

    # first volume
    pipeline1 = DummyPipeline(data=data[:, :3], n=1)
    ema._run(pipeline1)

    # second volume
    pipeline2 = DummyPipeline(data=data[:, :4], n=2)
    out = ema._run(pipeline2)

    assert out.shape == (2, 1)
    assert ema.filt.shape == (2, 1)


def test_SnormStep(sample_data):
    data_2d = sample_data.this_t_data[:, np.newaxis]

    pipeline = DummyPipeline(processed_tr=data_2d)
    step = SnormStep(save=False)

    res = step._run(pipeline)

    assert res.shape == data_2d.shape
    assert np.isclose(np.mean(res), 0.0, atol=1e-12)
    assert np.isclose(np.std(res), 1.0, atol=1e-12)