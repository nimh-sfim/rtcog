# Folder Description

This is the main folder for the software. Here you can find the following sub-folders:

* ```afni_lib```: a copy of realtime.py and receiver.py that contain all the functions for stabilishing the communication between AFNI realtime plugin and this software.

> __NOTE__: It may be nice to change the software to use whatever of these get distributed with AFNI in the afniInterfaceRT folder. That way if AFNI updates these functions, the program will be up-to-date. Yet, it would be good to have some automatic test to ensure any changes in the AFNI part, does not break this software.

* ```bin```: this folder contains the main executables, including ```rtcaps_matcher.py``` (which does realtime preprocessing and experience sampling), and ```online_trainSVRs.py``` (which train SVRs). It includes additional small programs to test each pre-processing step, sound, keyboard, etc.

* ```notebooks```: includes python notebooks used during the development of this code.

> __NOTE__: We need to check those, and ensure they are a 1-to-1 match with the final version of programs and that I did not left them half-way finalized.

* ```resources```: contains images and sounds used by ```rtcaps_matcher.py``` in order to generate the GUI that subjects experience during the experimental runs.

* ```rtcap_lib```: internal libraries with code for pre-processing, svr training, hit detection, etc. It is basically the core of the code.

* ```tests```: a series of notebooks to test individual pre-processing steps.

> __NOTE__: This folder and the notebooks folder should be merged at some point.


# How to do simulate a experiment from your laptop (no access to scanner)

This section describes how to simulate experiments without access to the scanner. This is very useful during software development, as all functionalities can be tested without having to request scanner time.

During an experiment, there are three different computers involved. Data flows in the following manner. First, the scanner (1) acquires images and sends them to the realtime system (2). For each incoming EPI image, the realtime system performs the following operations: alignment towards reference volume, estimation of head motion, extraction of data within a mask. Both motion parameters and extracted voxel-wise values within the mask are subsequently sent via TCP/IP to the experimental laptop (3). The experimental laptop takes incoming data, does additional pre-processing, and then drives the experiment GUI based on how that incoming data looks like a series of pre-defined templates.

![](../Documentation/Images/experimental_computers.png)

When developing this software, we will need to simulate the workings of these three systems, but using a single machine (our development machine). The rest of this section describes how to accomplish this process.

1. Open a terminal and create three distinct folders:

* __Scanner__: This folder contains testing data (available here). Once you create this folder, download the testing data and place it here.

* __Realtime__: This will be an empty folder, where AFNI realtime will save all incoming data.

* __Laptop__: This will be an empty folder, where rtCAPs software will save different output such as reports, trained classifiers, subject responses, etc.

2. Open three different Terminals in your laptop, and give them the following names:

* __Scanner__: you will use this window to simulate the scanner sending data to AFNI realtime
* __Realtime__: here you will start AFNI in realtime mode. It will take incoming data from the "fake" scanner, and after a few things sending on its way to the rtCAPs software.
* __Laptop__: here you will start the rtCAPs software.

2. On the __Scanner__ terminal, create a new empty directory, and make sure to copy sample datasets. To a minimum you should have an anatomical dataset, a short EPI dataset to use as reference for alignment, and then two additional long EPI datasets: one will be used for training the classifier and the second one to simulate a real experience sampling run.

 ![](../Documentation/Images/simulation_terminals.png)

3. On the __Realtime__ teminal, do the following:
    
* Create a new empty directory.
    
* Copy the 01_BringROIsToSubjectSpace.sh script here.
    
* Copy the Frontiers2013_CAPs.nii file here.
    
* Export the following variables

```bash
export AFNI_REALTIME_Registration=3D:_realtime
export AFNI_REALTIME_Base_Image=2
export AFNI_REALTIME_Graph=Realtime
export AFNI_REALTIME_MP_HOST_PORT=localhost:53214
export AFNI_REALTIME_SEND_VER=YES
export AFNI_REALTIME_SHOW_TIMES=YES
export AFNI_REALTIME_Mask_Vals=ROI_means
export AFNI_REALTIME_Function=FIM
```

* Start AFNI in realtime mode

```$ afni -rt```

> __NOTE__: Make sure you have the latest version of AFNI installed, as you will be using a data transfer option only available since 2020.

4. Simulate acquisition of anatomical dataset

On the __Scanner__ console, type:

```bash
rtfeedme Anat+orig
```

By the end of this step, you should have a new dataset (rt.__001+orig) that contains the anatomical data (but now in the realtime system)

5. Simular acquisition of the EPI reference dataset

On the __Scanner__ console, type:

```bash
rtfeedmd EPI_Reference+orig
```

By the end of this step, you should have a second dataset on __Realime__ (rt.__002+orig) that contains the EPI reference data (but now in the realtime system)

6. On the __Realime__ terminal, nun 01_BringROIsToSubjectSpace.sh as follows:

```bash
sh ./01_BringROIsToSubjectSpace.sh rt.__002+orig. rt.__001+orig. Frontier2013_CAPs.nii
```

This will generate a lot of new files, among the most important ones:

* EPIREF+orig: this will become our reference volume for realtime alignemnt.
* GMribbon_R4Feed.nii: this will be our mask for sending data to the laptop.
* Frontiers2013_R4Feed.nii: this will be our CAPs template aligned to the EPI data.

The last two files need to be transfered to the __Laptop__ directory.

7. Configure the realtime plugin for the rest of the experiment.

In the main AFNI window, click on Define Datamode --> Plugins --> RT Options

On the new window, ensure the following configurations:

* Registration = 3D: realtime
* Resampling = Quintic
* Reg Base = External Dataset
* External Dset = EPIREF+orig
* Base Image = 0
* NR = 1200 (Or as many volumes as you are expecting in the next run)
* Mask = GMribbon_R4Feed.nii
* Val to Send = All Data (light)

8. Start rtCAPs in pre-processing mode in the __Laptop__ terminal.

```bash 
python ../../rtcaps/bin/rtcaps_matcher.py \
        --mask GMribbon_R4Feed.nii \
        --nvols 1200 \
        -e preproc \
        --save_all \
        --out_dir ./ \
        --out_prefix training
```

9. Simulate acquisition of the traning run

    In the __Scanner__ console, type:

```bash
rtfeedme TrainingRun+orig
```

The data will be send to AFNI, who in turn will do motion correction (towards the EPI reference dataset), and then send the values of each voxel in the GMribbon mask to the rtCAPs program that is listening by default on port 53214. By the end of this step, in the __Laptop__ folder you should have the following files:

* ```$prefix_Options.json```: record of all the options.
* ```$prefix.Motion.1D```: motion estimates.
* ```$prefix.Zscore.nii```: final per-TR activity map?
* ```$prefix.pp_EMA.nii```: data following the EMA step.
* ```$prefix.pp_iGLM.nii```: data following the incremental GLM step.
* ```$pretix.pp_iGLM_$regressor.nii```: fitting (beta value) of each nuisance regressor.
* ```$prefix.pp_LPfilter.nii```: data following the low pass filtering step.
* ```$prefix.pp_Smotth.nii```: data following the spatial smoothing step.

10. Train the SVR

For that, in the __Laptop__ terminal, you should run:

```bash
 python ../../rtcaps/bin/online_trainSVRs.py \
        -d ./training.pp_Zscore.nii \
        -m ./GMribbon_R4Feed.nii \
        -c ./Frontier2013_CAPs_R4Feed.nii \
        -o ./ \
        -p training_svr
```

This will generate the following additional files in the __Laptop__ folder:

* ```training_svr.pkl```: trained SVRs (needed for the rest of the experimental runs)
* ```training_svr_training_vols.csv```: ?
* ```training_svr_lm_R2.csv```: R2 for linear regerssion on training data
* ```training_svr_lm_z_labels.csv```: TR-by-TR labels of SVRs (after Z-scoring) 
* ```training_svr.png```: static summary of SVR traning
* ```training_svr.html```: dynamic summary of SVR traning

Here is an example of the static training report

![Sample of Traning SVR Static Report](../Documentation/Images/training_svr.png)

11. Start rtCAPs to deal with a real Experience Sampling Run

For that, on the __Laptop__ terminal run the following:

```
# Pre-processing of Traning Run

Despite the lack of realtime experiment control during the training run (the subject will see the crosshair in the screen all the time), we still run software in the laptop during this run for two purposes:

1. The GUI the subject sees is provided by the laptop, and in that way, it matches the same GUI subjects will see during the experimental run.

2. We need to pre-process the incoming data in the same way, as the experimental data will be pre-processed curing real experimental runs. At the end of this run, we need a fully pre-processed dataset in the laptop, ready to be input to the SVR traning.

The program that will do these operations for us is:

```$PRJDIR/rtcaps/bin/rtcaps_matcher.py```

When operated for the purpose of training, you need to provide at least the following parameters:

* ```--mask```: This will be the GM ribbon mask already in EPI space. This mask is generated by script ```01_BringROIsToSubjectSpace.sh``` which needs to be run at the scanner console at the completion of the anatomical and the EPI reference scan. This file then needs to be transfered to the laptop.

* ```--nvols```: Number of expected acquisitions for the training run. 

* ```-e preproc```: This indicates the program that this is a training run, and therefore it must only perform pre-processing, and nothing else.

* ```--save_all```: Save all possible outputs

* ```--out_dir```: Output directory for saving results

* ```--out_prefix```: File prefix for all output files.

```bash
python ./rtcaps_matcher.py --mask ../../PrcsData/TECH07/D03_RTSim_3iso_Set01/GMribbon_R4Feed.nii --nvols 500 -e preproc --save_all --out_dir ./temp --out_prefix training
```

Outputs:

** ```$prefix.Motion.1D```: motion estimates.
** ```$prefix.Zscore.nii```: final per-TR activity map?
** ```$prefix.pp_EMA.nii```: data following the EMA step.
** ```$prefix.pp_iGLM.nii```: data following the incremental GLM step.
** ```$pretix.pp_iGLM_$regressor.nii```: fitting (beta value) of each nuisance regressor.
** ```$prefix.pp_LPfilter.nii```: data following the low pass filtering step.
** ```$prefix.pp_Smotth.nii```: data following the spatial smoothing step.
** ```$prefix.svrscores```: numpy array with the SVR scores per TR. There will be one score per CAP (e.g. per pre-trained SVR)
** ```$prefix.hits```: information about hits.
** ```$prefix.Hit_```: 
# Training Classifiers

# Experimental Rest run with Experience Sampling

This is the core of the experiment. Here, the software will present a crosshair in the center of the screen, and in the background it will both pre-process the incoming data, but also, when there is a hit, it will fire an experience sampling probe. 

The minimum parameters to ensure this mode of operation are:

** ```--mask```: GM Ribbon mask in the same space as incoming EPI data.
** ```--nvols```: Number of expected acquisitions
** ```-e esam```: Ensure the software runs in Experience SAMpling mode.
** ```--svr_path```: Path to pre-trained Support Vector Regression Machines (generated by the step described above)
** ```--svr_win_activate```: activate windowing of individual volumes prior to hit estimation (deafult = False).
** ```--out_dir```: output directory
** ```--fscreen```: full screen mode.
