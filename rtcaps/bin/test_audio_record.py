import pyaudio
import wave
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rtcap_lib.recorder import Recorder

rec_path = '/data/test.wav'
rec = Recorder(channels=1)
with rec.open(rec_path,'wb') as rec_file:
        rec_file.start_recording()
        time.sleep(5.0)
        rec_file.stop_recording()
