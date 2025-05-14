import sys
import os
import os.path as osp
import time

from rtcap_lib.recorder import Recorder
from rtcaps.config import DATA_DIR

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

rec_path = osp.join(DATA_DIR, 'test.mp3')
rec = Recorder(channels=1)
with rec.open(rec_path,'wb') as rec_file:
        rec_file.start_recording()
        time.sleep(5.0)
        rec_file.stop_recording()
