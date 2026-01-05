import os.path as osp
import pytest
from unittest.mock import patch, mock_open

from rtcog.utils.options import Options
from rtcog.paths import CONFIG_DIR

def test_init():
    config = {'exp_type': 'basic', 'nvols': 100}
    options = Options(config)
    assert options.exp_type == 'basic'
    assert options.nvols == 100


@patch('rtcog.utils.options.Options.parse_cli_args')
def test_from_cli(mock_parse):
    mock_parse.return_value = {'exp_type': 'basic'}
    options = Options.from_cli()
    assert isinstance(options, Options)
    assert options.exp_type == 'basic'


@patch('builtins.open', new_callable=mock_open)
@patch('yaml.safe_dump')
def test_save_config(mock_dump, mock_file):
    config = {'exp_type': 'basic', 'nvols': 100, 'out_dir': '/tmp', 'out_prefix': 'test'}
    options = Options(config)
    options.save_config()
    mock_file.assert_called_once_with('/tmp/test_Options.yaml', 'w')
    mock_dump.assert_called_once()


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
        'steps': [{'name': 'EMA', 'enabled': True, 'save': False, 'ema_thr': 0.98},
                {'name': 'iGLM',
                    'enabled': True,
                    'save': False,
                    'num_polorts': 2,
                    'iGLM_motion': True},
                {'name': 'kalman', 'enabled': True, 'save': False, 'n_cores': 10},
                {'name': 'smooth', 'enabled': True, 'save': False, 'fwhm': 4.0},
                {'name': 'snorm', 'enabled': True, 'save': False},
                {'name': 'windowing',
                    'enabled': True,
                    'save': False,
                    'win_length': 4}],
        'no_action': False,
        'no_proc_chair': False,
        'fullscreen': False,
        'screen': 1,
        'q_path': 'questions_v1',
        'snapshot': False,
        'test_latency': False,
        'matching': {'match_method': 'mask',
                    'match_start': 100,
                    'vols_noaction': 45},
        'hits': {'nconsec_vols': 2,
                'nonline': 1,
                'do_mot': True,
                'mot_thr': 0.2}
    }
    assert opts.__dict__ == expected_dict


def test_missing_required_args_raises():
    with pytest.raises(SystemExit):
        Options.parse_cli_args(["-e", "basic"])


def test_esam_missing_matching_section(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("exp_type: esam\n")

    mask = tmp_path / "mask.nii"
    mask.touch()

    with pytest.raises(SystemExit):
        Options.parse_cli_args(
            [
                "--config", str(config),
                "--mask", str(mask),
                "--nvols", "100",
                "--out_dir", "/tmp",
                "--out_prefix", "test"
            ]
        )

    err = capsys.readouterr().err
    assert "'matching' section is required for esam experiment" in err


def test_esam_missing_required_args(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("""
        exp_type: esam
        matching:
           match_method: mask
        """)

    mask = tmp_path / "mask.nii"
    mask.touch()

    with pytest.raises(SystemExit):
        Options.parse_cli_args(
            [
                "--config", str(config),
                "--mask", str(mask),
                "--nvols", "100",
                "--out_dir", "/tmp",
                "--out_prefix", "test"
            ]
        )

    err = capsys.readouterr().err
    assert "The following arguments are required:" in err
    assert "--hit_thr" in err
    assert "--match_path" in err


if __name__ == "__main__":
    pytest.main()