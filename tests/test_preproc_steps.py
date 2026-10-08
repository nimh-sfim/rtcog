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
    ema = EMAStep(save=False, alpha=0.98, Nv=2, Nt=3)

    out = ema._run(pipeline)

    assert out.shape == (2, 1)
    assert ema.filt.shape == (2, 1)


def test_EMAStep_second():
    data = np.array([[1, 2, 3, 4],
                     [4, 5, 6, 7]])

    ema = EMAStep(save=False, alpha=0.98, Nv=2, Nt=4)

    # first volume
    pipeline1 = DummyPipeline(data=data[:, :3], n=1)
    ema._run(pipeline1)

    # second volume
    pipeline2 = DummyPipeline(data=data[:, :4], n=2)
    out = ema._run(pipeline2)

    assert out.shape == (2, 1)
    assert ema.filt.shape == (2, 1)


def test_unsaved_step_snapshot_is_empty():
    ema = EMAStep(save=False, alpha=0.98, Nv=2, Nt=4)

    assert ema.snapshot() == {}


def test_snorm_step_normalizes_synthetic_volume(synthetic_fmri_data):
    data_2d = synthetic_fmri_data.data[..., 0][
        synthetic_fmri_data.mask
    ][:, np.newaxis]
    assert np.std(data_2d) > 0

    pipeline = DummyPipeline(processed_tr=data_2d)
    step = SnormStep(save=False, Nv=data_2d.shape[0], Nt=1)

    res = step._run(pipeline)
    expected = (data_2d - np.mean(data_2d)) / np.std(data_2d)

    assert res.shape == data_2d.shape
    assert np.isclose(np.mean(res), 0.0, atol=1e-12)
    assert np.isclose(np.std(res), 1.0, atol=1e-12)
    np.testing.assert_allclose(res, expected, rtol=0, atol=1e-12)
