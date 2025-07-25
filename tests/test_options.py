import os.path as osp
import pytest
import pprint

from rtfmri.utils.options import Options
from rtfmri.paths import CONFIG_DIR

def test_from_yaml():
    path = osp.join(CONFIG_DIR, 'default_config.yaml')
    opts = Options.from_yaml(path)

    expected_dict = {
        'debug': False,
        'silent': False,
        'tcp_port': 53214,
        'show_data': False,
        'tr': 1.0,
        'save_orig': False,
        'discard': 10,
        'steps': [{'name': 'EMA', 'enabled': True, 'save': False},
                {'name': 'iGLM',
                    'enabled': True,
                    'save': False,
                    'iGLM_polort': 2,
                    'iGLM_motion': True},
                {'name': 'kalman', 'enabled': True, 'save': False, 'n_cores': 10},
                {'name': 'smooth', 'enabled': True, 'save': False, 'fwhm': 4.0},
                {'name': 'snorm', 'enabled': True, 'save': False},
                {'name': 'windowing',
                    'enabled': True,
                    'save': False,
                    'win_length': 4}],
        'no_gui': False,
        'no_proc_chair': False,
        'fullscreen': False,
        'screen': 1,
        'q_path': 'questions_v1',
        'snapshot': False,
        'test_latency': False,
        'matching': {'match_method': 'mask',
                    'match_start': 100,
                    'vols_noqa': 45,
                    'do_win': True,
                    'win_length': 4},
        'hits': {'nconsec_vols': 2,
                'hit_method': 'method01',
                'do_mot': True,
                'mot_thr': 0.2}
    }
    assert opts.__dict__ == expected_dict


if __name__ == "__main__":
    pytest.main()