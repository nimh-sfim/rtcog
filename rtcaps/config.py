import os.path as osp

CODE_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = osp.abspath(osp.join(CODE_DIR, '..'))
RESOURCES_DIR = osp.join(CODE_DIR, 'resources/')
DATA_DIR = osp.join(CODE_DIR, "tests/data/")

SIMULATION_DIR = osp.join(ROOT_DIR, 'Simulation')
LAPTOP_DIR = osp.join(SIMULATION_DIR, 'Laptop/')
REALTIME_DIR = osp.join(SIMULATION_DIR, 'Realtime/')
SCANNER_DIR = osp.join(SIMULATION_DIR, 'Scanner/')
OUTPUT_DIR = osp.join(SIMULATION_DIR, 'outputs')

CAP_labels = ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']
CAP_indexes = [25, 4, 18, 28, 24, 11, 21]