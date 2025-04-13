import os.path as osp

CODE_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = osp.abspath(osp.join(CODE_DIR, '..'))
print(ROOT_DIR)

SIMULATION_DIR = osp.join(ROOT_DIR, 'Simulation')
LAPTOP_DIR = osp.join(SIMULATION_DIR, 'Laptop/')
REALTIME_DIR = osp.join(SIMULATION_DIR, 'Realtime/')
SCANNER_DIR = osp.join(SIMULATION_DIR, 'Scanner/')
OUTPUT_DIR = osp.join(SIMULATION_DIR, 'outputs')

# TODO: add the dirs to git (empty)


# import logging

# def setup_logger():
#     logger = logging.getLogger('online_preproc')

#     # Check if the logger has any handlers already, to avoid adding duplicate handlers
#     if not logger.hasHandlers():
#         print('++ LOGGER: setting')
#         logger.setLevel(logging.INFO)

#         log_fmt = logging.Formatter('[%(levelname)s - %(filename)s]: %(message)s')

#         # File Handler (overwriting the log file each time)
#         file_handler = logging.FileHandler("online_preproc.log", mode='w')  # 'w' mode overwrites the log file
#         file_handler.setFormatter(log_fmt)

#         # Stream Handler (for console output)
#         stream_handler = logging.StreamHandler()
#         stream_handler.setFormatter(log_fmt)

#         # Add handlers to the logger
#         logger.addHandler(file_handler)
#         logger.addHandler(stream_handler)
#     print(f'++ LOGGER HANDLERS: {logger.handlers}')  # Check the list of handlers added to the logger

#     return logger




# Case 1
#TRAIN_Path        = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/pb04.TECH07_Run01_Training.nii'
#TRAIN_Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/pb04.TECH07_Run01_Training.Motion.1D'
#Mask_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/GMribbon_R4Feed.nii'
#CAPs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Frontier2013_CAPs_R4Feed.nii'
#SVRs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/Offline_SVRs.pkl'
#TEST_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/pb04.TECH07_Run01_Testing.nii'
#TEST_Motion_Path  = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/pb04.TECH07_Run01_Testing.Motion.1D'
#OUT_Dir           = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/'
#Data_Path        = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Training.nii'
#Data_Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Training.Motion.1D'
#Data_Path        = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Testing.nii'
#Data_Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH07/D01_RT_Run01/TECH07_Run01_Testing.Motion.1D'
#TRAIN_NVols_Discard = 0
#CAP_indexes = [25,4,18,28,24,11,21]
#CAP_labels  = ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']

# Case 2
#TRAIN_Path        = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/TECH08_3iso_Set01_Training+orig.HEAD'
#TRAIN_Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/TECH08_3iso_Set01_Training.Motion.1D'
#Mask_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/GMribbon_C_R4Feed.nii'
#CAPs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/Frontier2013_CAPs_R4Feed.nii'
#SVRs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/Offline_SVRs.pkl'
#TEST_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/TECH08_3iso_Set01_Testing+orig.HEAD'
#TEST_Motion_Path  = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/TECH08_3iso_Set01_Testing.Motion.1D'
#OUT_Dir           = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set01_Data/'
#TRAIN_NVols_Discard = 0
#CAP_indexes = [25,4,18,28,24,11,21]
#CAP_labels  = ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']

# Case 3
# TRAIN_Path        = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/rt01.TECH08_3iso_Set02_Training.volreg+orig.HEAD'
# TRAIN_Motion_Path = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/TECH08_3iso_Set02_Training.Motion.1D'
# Mask_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/GMribbon_C_R4Feed.nii'
# CAPs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/Frontier2013_CAPs_R4Feed.nii'
# SVRs_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/TECH08_3iso_Set02_Training.volreg.Offline_SVRs.pkl'
# TEST_Path         = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/rt01.TECH08_3iso_Set02_Testing.volreg+orig.HEAD'
# TEST_Motion_Path  = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/TECH08_3iso_Set02_Testing.Motion.1D'
# OUT_Dir           = '/data/SFIMJGC_HCP7T/PRJ_rtCAPs/PrcsData/TECH08/D01_3iso_Set02_Data/'
# TRAIN_OUT_Prefix  = 'TECH08_3iso_Set02_Training'
# TEST_OUT_Prefix   = 'TECH08_3iso_Set02_Testing'
# TRAIN_NVols_Discard = 0
# CAP_indexes = [25,4,18,28,24,11,21]
# CAP_labels  = ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']