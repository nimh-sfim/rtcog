# Preprocessing of Training Run
python ./rtcaps_matcher.py --mask ../../PrcsData/TECH07/D03_RTSim_3iso_Set01/GMribbon_R4Feed.nii --nvols 500 -e preproc --save_all --out_dir ./temp --out_prefix training

# Train SVRs
python ./online_trainSVRs.py -d ./temp/training.pp_Zscore.nii -m ./temp/GMribbon_R4Feed.nii -c ./temp/Frontier2013_CAPs_R4Feed.nii -o ./temp -p training_svr

# Experimental Run
python ./rtcaps_matcher.py --mask ../../PrcsData/TECH07/D03_RTSim_3iso_Set01/GMribbon_R4Feed.nii --nvols 1000 -e esam --svr_path ../../PrcsData/TECH07/D03_RTSim_3iso_Set01/TECH07_3iso_Set01_Train.pp_Zscore.SVR.pkl --svr_win_activate --out_dir ./temp/ --fscreen
