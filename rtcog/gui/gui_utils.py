import sys
import os
import json
import os.path as osp

from psychopy import core, gui, data
from psychopy.hardware import keyboard
from psychopy import prefs

if os.environ.get("READTHEDOCS") != "True":
    prefs.hardware['keyboard'] = 'pygame'

from rtcog.utils.core import get_logger
from rtcog.paths import RESOURCES_DIR

log = get_logger()

def validate_likert_questions(q_path):
    """Ensure the questions provided are valid."""
    if not q_path:
            log.error('Path to Likert questions was not provided. Program will exit.')
            sys.exit(-1)
    if not osp.isfile(q_path): # If not file, assume in RESOURCES_DIR
        fname = q_path + ".json" if not q_path.endswith(".json") else q_path 
        q_path = osp.join(RESOURCES_DIR, fname)
    try:
        with open(q_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.error(f'The question file at {q_path} is not a valid JSON.')
        sys.exit(-1)
    except Exception as e:
        log.error(f'Error loading questions at {q_path}: {e}')
        sys.exit(-1)

def get_avail_keyboards():
    available_keyboards = keyboard.getKeyboards()
    available_keyboards_labels = []
    for kb in available_keyboards:
        if kb['product'] == '': 
            available_keyboards_labels.append('Laptop Keyboard')
        else:
            available_keyboards_labels.append(kb['product'])
    return available_keyboards, available_keyboards_labels

def get_experiment_info(opts):
    available_keyboards, available_keyboards_labels = get_avail_keyboards()
    expInfo = {
        'prefix':      opts.out_prefix,
        'out_dir':     opts.out_dir,
        'keyboard':    available_keyboards_labels,
        'screen':      ['Laptop','External'],
        'fullScreen':  ['Yes','No'],
        'leftKey':     '3',
        'rightKey':    '1',
        'acceptKey':   '2',
        'triggerKey':  't'
    }
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title='rtCAPs Thought Sampling')
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    expInfo['date'] = data.getDateStr()
    return expInfo

