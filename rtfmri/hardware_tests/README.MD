# Hardware tests

This directory contains scripts to test essential hardware components before a scanning session.

## `test_scanner_latency.py`

Measure the latency between scanner triggers (via `t` keystrokes) and the time the software receives associated data.

Usage:

```python
python test_scanner_latency.py --out_dir <output_directory> --out_prefix <output_prefix>
```

- After starting the script, click into the GUI window so it has focus.
- The timing of`t` keystrokes and data receipt will be logged and saved.

## `test_keystrokes.py`

Verify that keystrokes are correctly detected by the system.

Usage:

```python
python test_keystrokes.py --out_dir <output_directory> --out_prefix <output_prefix>
```

- After starting the script, click into the GUI window so it has focus.
- Press the `t` key to simulate scanner pulses.
- All keystrokes will be logged and saved.
