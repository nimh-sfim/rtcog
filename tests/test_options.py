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
        'save_orig': False,
        'discard': 10,
        'steps': [{'name': 'EMA', 'enabled': True, 'save': False, 'alpha': 0.98},
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
        'fullscreen': False,
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
    with pytest.raises(ValueError, match="Please specify a config file using --config/-c."):
        Options.parse_cli_args(["-e", "basic"])


def test_basic_required_args_can_all_come_from_yaml(tmp_path):
    mask = tmp_path / "mask.nii"
    mask.touch()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
        exp_type: basic
        mask_path: {mask}
        nvols: 100
        out_dir: {out_dir}
        out_prefix: test
        """
    )

    parsed = Options.parse_cli_args(["--config", str(config)])

    assert parsed["exp_type"] == "basic"
    assert parsed["mask_path"] == str(mask)
    assert parsed["nvols"] == 100
    assert parsed["out_dir"] == str(out_dir)
    assert parsed["out_prefix"] == "test"


def test_save_orig_cli_overrides_yaml_false(tmp_path):
    mask = tmp_path / "mask.nii"
    mask.touch()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
        exp_type: basic
        mask_path: {mask}
        nvols: 100
        out_dir: {out_dir}
        out_prefix: test
        save_orig: false
        """
    )

    parsed = Options.parse_cli_args(
        ["--config", str(config), "--save_orig"]
    )

    assert parsed["save_orig"] is True


def test_omitted_cli_booleans_preserve_yaml_true(tmp_path):
    mask = tmp_path / "mask.nii"
    mask.touch()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
        exp_type: basic
        mask_path: {mask}
        nvols: 100
        out_dir: {out_dir}
        out_prefix: test
        debug: true
        silent: true
        show_data: true
        save_orig: true
        auto_save: true
        no_action: true
        fullscreen: true
        snapshot: true
        test_latency: true
        """
    )

    parsed = Options.parse_cli_args(["--config", str(config)])

    for key in (
        "debug",
        "silent",
        "show_data",
        "save_orig",
        "auto_save",
        "no_action",
        "fullscreen",
        "snapshot",
        "test_latency",
    ):
        assert parsed[key] is True


def test_cli_long_options_match_yaml_keys(tmp_path):
    yaml_mask = tmp_path / "yaml-mask.nii"
    yaml_mask.touch()
    cli_mask = tmp_path / "cli-mask.nii"
    cli_mask.touch()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
        exp_type: basic
        mask_path: {yaml_mask}
        nvols: 100
        out_dir: {out_dir}
        out_prefix: test
        fullscreen: false
        test_latency: false
        """
    )

    parsed = Options.parse_cli_args(
        [
            "--config", str(config),
            "--mask_path", str(cli_mask),
            "--fullscreen",
            "--test_latency",
        ]
    )

    assert parsed["mask_path"] == str(cli_mask)
    assert parsed["fullscreen"] is True
    assert parsed["test_latency"] is True


def test_legacy_cli_aliases_remain_supported(tmp_path):
    yaml_mask = tmp_path / "yaml-mask.nii"
    yaml_mask.touch()
    cli_mask = tmp_path / "cli-mask.nii"
    cli_mask.touch()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
        exp_type: basic
        mask_path: {yaml_mask}
        nvols: 100
        out_dir: {out_dir}
        out_prefix: test
        fullscreen: false
        test_latency: false
        """
    )

    parsed = Options.parse_cli_args(
        [
            "--config", str(config),
            "--mask", str(cli_mask),
            "--fscreen",
            "--latency",
        ]
    )

    assert parsed["mask_path"] == str(cli_mask)
    assert parsed["fullscreen"] is True
    assert parsed["test_latency"] is True


def test_esam_required_args_can_all_come_from_yaml(tmp_path):
    mask = tmp_path / "mask.nii"
    mask.touch()
    match = tmp_path / "templates.npz"
    match.touch()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
        exp_type: esam
        mask_path: {mask}
        nvols: 100
        out_dir: {out_dir}
        out_prefix: test
        match_path: {match}
        hit_thr: 0.5
        matching:
          match_method: mask
        """
    )

    parsed = Options.parse_cli_args(["--config", str(config)])

    assert parsed["match_path"] == str(match)
    assert parsed["hit_thr"] == 0.5


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
