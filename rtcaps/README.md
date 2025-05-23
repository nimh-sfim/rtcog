# Folder Description

This is the main folder for the software. Here you can find the following sub-folders:

* ```bin```: this folder contains the main executables, including ```rtcaps_matcher.py``` (which does realtime preprocessing and experience sampling), and ```online_trainSVRs.py``` (which trains SVRs). It includes additional small programs to test each pre-processing step, sound, keyboard, etc.

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

<u>1. Open three different Terminals on your laptop in the `Simulation/` directory, then `cd` to the following directories:</u>

* __Scanner__: you will use this window to simulate the scanner sending data to AFNI realtime
* __Realtime__: here you will start AFNI in realtime mode. It will take incoming data from the "fake" scanner, and after a few things sending on its way to the rtCAPs software.
* __Laptop__: here you will start the rtCAPs software.

 ![](../Documentation/Images/simulation_terminals.png)

<u>2. Download the sample data</u>

* Enter the empty __Scanner__ folder.
* Download sample datasets to the __Scanner__ folder.

To a minimum you should have an anatomical dataset, a short EPI dataset to use as reference for alignment, and then two additional long EPI datasets: one will be used for training the classifier and the second one to simulate a real experience sampling run.

<u>3. Go to the __Realtime__ teminal:</u>
    
* Enter the empty __Realtime__ folder.
* Copy the 01_BringROIsToSubjectSpace.sh script here.
* Copy the Frontiers2013_CAPs.nii file here.
* Start up afni realtime: 
```bash
sh ../../rtcaps/bin/startup_afnirt.sh --reference
```
> __NOTE__: Make sure you have the latest version of AFNI installed, as you will be using a data transfer option only available since 2020.

<u>4. Simulate acquisition of anatomical dataset</u>

On the __Scanner__ console, type:

```bash
rtfeedme Anat+orig
```

By the end of this step, you should have a new dataset (rt.__001+orig) that contains the anatomical data (but now in the realtime system) in the __Realtime__ folder.

<u>5. Simulate acquisition of the EPI reference dataset</u>

On the __Scanner__ console, type:

```bash
rtfeedme EPI_Reference+orig
```

By the end of this step, you should have a second dataset on __Realime__ (rt.__002+orig) on the __Realtime__ folder that contains the EPI reference data (but now in the realtime system)

<u>6. Pre-process Anatomical and bring masks to EPI Reference space</u>

* Go to the __Realtime__ terminal
* Run `01_BringROIsToSubjectSpace.sh` as follows:

```bash
sh ./01_BringROIsToSubjectSpace.sh \
       rt.__002+orig. \
       rt.__001+orig. \
       Frontier2013_CAPs.nii
```

This will generate a lot of new files. The key ones moving forward are:

* ```EPIREF+orig```: this will become our reference volume for realtime alignemnt.
* ```GMribbon_R4Feed.nii```: this will be our mask for sending data to the laptop.
* ```Frontiers2013_R4Feed.nii```: this will be our CAPs template aligned to the EPI data.

The last two files need to be transfered (i.e., copied) to the __Laptop__ directory.

```bash
$cp ${REALTIME_FOLDER}/GMribbon_R4Feed.nii ${LAPTOP_FOLDER}
$cp ${REALTIME_FOLDER}/Frontiers2013_R4Feed.nii ${LAPTOP_FOLDER}
```

<u>7. Configure the realtime plugin for the rest of the experiment.</u>

- Kill your current AFNI rt process
- In the __Realtime__ terminal, run:
```bash
sh ../../rtcaps/bin/startup_afnirt.sh
```

<u>8. Start rtCAPs in pre-processing mode in the __Laptop__ terminal.</u>

```bash 
python ../../rtcaps/bin/rtcaps_matcher.py \
        --mask GMribbon_R4Feed.nii \
        --nvols 1200 \
        -e preproc \
        --save_all \
        --out_dir ./ \
        --out_prefix training
```

<u>9. Simulate acquisition of the training run</u>

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

> __NOTE__: Instead of manually configuring options in the future, you can simply supply the json file:
>
>`python ../../rtcaps/bin/rtcaps_matcher.py --config path/to/$prefix_Options.json`

<u>10. Train the SVR</u>

Select the CAPs of interest and create txt file with labels:

```bash
3dTcat -prefix Templates_R4Feed.nii Frontier2013_CAPs_R4Feed.nii"[25, 4, 18, 28, 24, 11, 21]"
echo "VPol,DMN,SMot,Audi,ExCn,rFPa,lFPa" >> template_labels.txt
```

Go to the __Laptop__ terminal and run:

```bash
 python ../../rtcaps/bin/online_trainSVRs.py \
        -d ./training.pp_Zscore.nii \
        -m ./GMribbon_R4Feed.nii \
        -t ./Frontier2013_CAPs_R4Feed.nii \
        -l ./template_labels.txt \
        -o ./ \
        -p training_svr \
        --no_lasso
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

<u>11. Start rtCAPs to deal with a real Experience Sampling Run</u>

For that, on the __Laptop__ terminal run the following:

```bash
python ../../rtcaps/bin/rtcaps_matcher.py
       --nvols 1200 \
       --mask GMribbon_R4Feed.nii \
       --out_dir ./ \
       --out_prefix esam \
       -e esam \
       --svr_path training_svr.pkl \
       --svr_win_activate
```
***
***


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
