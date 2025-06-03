import sys
import os.path as osp
from playsound import playsound

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
from config import RESOURCES_DIR

sound_path = osp.join(RESOURCES_DIR, 'bike_bell.wav')
playsound(sound_path)