# rtCAPs

This project contains the code to perform realtime fMRI experiments in the following form:
1. The system monitors brain activity on a TR-by-TR basis
2. If the subject brain looks like one of a subset of templates, then the program fires a questionaire
3. The program continues monitoring the brain in the background.

> NOTE: The goal was to ensure that all these things can happen while running from Javier's laptop and with a TR of 1s.


First, let's have a look at the directories in here:

* Dashboards: seems to only include .png files. I believe thos may be outcomes, but not sure which part of the code generates them.
* afni_plugin: I have to use a special version of the afni plugin so that I could do some specific configuration. I need to check and add info here.
* deprecated: hopefully something I no longer us, and I won't have to worry about it.
* emails: old documentaiton, but this may be very useful fot catchin up.
* OpenNFT: this is the code from VanDerV1lle. We took some methods from here (we will definitely give credit)
* Others: templates for CAPs (provided by Lui), templates for the Smith networks (2009), GM ribbon and few others that I believe I must use in the code
* Papers: some papers of interest that I will need to read again.
* PrcsData: all data acquired for this project so far... there are some technical and pilot data... will need to separate them.
* Protocol: contains document used to ammend the M-93-XXX protocol
* Psychopy: I believe I use some of this to construct the questionaries, but not sure why I have a full distribution here
* Realtime: This is Vinai's software, I think I only used it for inspiration, but have to make sure.
* resources: it is empty... not sure why it was there.
* rtcaps: the core of the code
* ScanNotes: infor about techincal scans used for development.
* Scripts: not sure how this differs from the mani code
* TMP: No idea
* TraniningMaterials: documents to show subjects piror to entering the scanner.

***
# A Typical Scanning Session

![](./Documentation/Images/scan_session.png)

Before subjects enter the scanner, the experimenter would provide subjects with instructions about the experiment and run a few samples of how things will work inside the scanner (Introspection Training).

Next, subjects will be places in the scanner bore and:

* Pulse Oximeter will be placed on the left index finger
* Respiratory Belt will be placed on the abdomen
* Response Box/Joistick will be provided in the right hand
* Optoacoustic headphones will be put in place (over ear protection)
* Optoacoustic microphone will be place near the mouth
* Place mirror so subjects can read the projection screen.

Future options:

* Eye tracker
* MRI-Compatible High Resolution Camera (to record body motion and/or facial expressions)
* MRI-Compatible EEG

During each scanning session, the following scans will take place:

a. 3D Localizer
b. High Resolution T1
c. High Order Shimming (required EPI prescription)
d. Short 5vols EPI Reference scan (maybe blip up/blip down)

At this point, the experimenter should run the first pre-processing script (01_BringROIsToSubjectSpace.sh):

```bash
cd EXXXX
sh ./01_BringROIsToSubjectSpace.sh EPIREFFILE ANATREF CAPS_FILE
```

INPUTS:

* EPIREFILE will be the scan created by the AFNI realtime in step (d)
* ANATREF will be the scan created by the AFNI realtime in step (b)
* CAPS_FILE will be the local copy of Frontiers2013_CAPs.nii in the local directory (and previously copied as described in Section X below)
* [MNITEMPLATE]: If none provided, then by default it uses MNI152_2009_template_GM.nii.gz. I believe for this experiment we were using the default value.

OUTPUTS:

* GMribbon_R4Feed.nii: This is the mask that will be used for sending data in the AFNI realtime plugin. This GM ribbon is adapted from MNI_Glasser_HCP_v1.0.nii.gz.

* Frontiers2013_CAPs_R4Feed.nii: This are the templates aligned to the EPI space of this subject and on the same spatial resolution. 

* EPIREF+orig: This file provides the reference space for the rest of the experiment. It will be provided as the "Ext. Dset" in the AFNI realtime plugin (see below).

* Spatial Tranformation matrices: Need to run to see what exactly it produces.

> NOTE: GMribbon_R4Feed.nii and Frontiers2013_CAPs_R4Feed.nii will need to be transfered to the experimental laptop, as this mask is used on the other end of the communication to reconstruct the actual location of each datapoint. Instructions on how to perform this transfer are provided when the script finishes.








***

# Overview of what the Software does

![](./Documentation/Images/software_overview.png)
***

# Targeted CAPs

In July 2019, Xiao Liu (former post-doctoral fellow at Jeff Dyun group) was kind to share with me the CAPs template assocuated with his Frontiers 2013 paper. We provided me with the template for the 30 CAPs that were originally described on his work. Many of these CAPs are highly similar, so for our initial evaluation we decided to only use a subset of the CAPs (see figure below)

![](./Documentation/Images/selected_caps.png)

***

# Experimental Hardware Setup

Three different computers are involved in this experimental setup:

* Scanner Console
* Realtime System (AFNI)
* Experimental Laptop

All computers need to be in the intranet of the scanner (see instructions below)

In addition to the computers, other important pieces of hardware include:

* Audio Delivery System: we will use the noise cancellation headphones (OptoAcoustics)
* Audio Recording System: we will use the noise cancellation microphone (OptoAcoustics)
* Video Delivery System: we will use the screen at the back of the scanner + mirror.

Before we can run the experiment, several configuration steps are required:

### 1. AUDIO DELIERY SETUP

* To pass audio from the laptop to the OptoAcostics system use a headphones-like cable that goes from the laptop headphones connection to line 1 in the back of the optoacoustic box

* Ensure Line 1 volume (in OptoAcoustic box) is to the maximum

* In the laptop ensure Sound Output is set to External Headphones

    ![](./Documentation/Images/audio_input_config.png)

### 2. AUDIO RECORDING SYSTEM

* In the OptoAcoustics Box, ensure the following options/buttons are set correctly:
    * FOMRI Noise Canceller Button ON (Green)
    * MONITOR FORMI ON (Green)
    * Connect USB-A port from OptoAcoustics (USB OUT in back) to any USB-C port in the laptop using the small adaptors (Q)

    ![](./Documentation/Images/optoacustics_back.png)

    * Self Hearing Knob in the middle
    * Speaker Knob in the middle
* In the laptop set Audio Input to USB Audio CODEC:
    ![](./Documentation/Images/audio_output_config.png)

> NOTE: Don’t make any of the audio go through the CalDigital DOCK as it creates some sort of loop with the input and output of sound.

### 3. SCREEN / PROJECTION SETUP

* On the DL Matrix, ensure DVI is the input to LCD1, LCD2 and MRILCD
* Switch on the MRILCD and LCD2 screens
* Connect the MINI DP to the Computer directly using the StarTech.com adapter
* Ensure the LCD/BOLD Screens are the primary monitors (for Full Screen to work)

> NOTE: Do not connect the monitor through the dock as it causes issues with intermittent disconnection.

***

# Experimental Software Setup

### 1. Setup the correct AFNI Realtime Plugin

The AFNI realtime system has a basic version of the AFNI - realtime plugin. For this experiments we need a modified version. Vinai has a setup at the scanner so that both plugins can be present, but only one gets fired for a given scanning session. Need to ask Vinai to remind me what link needs to be changes for this to take effect.

Our AFNI realtime plugin comes with one additional option named (light all data). This option was created for performance improvements. By default the AFNI plugin allows the following data forwarding options:

* Motion Only: only sends the six motion parameters
* ROI Means: send the mean of all voxels in each ROI
* All Data: send the data for all voxels in the ROI. The way this is implemented, AFNI does not send one value per voxel, but actually eight values: voxel_id, x, y, z, i, j, k and value. This was a tremendous overload, and it was a performance issue.

With the modified plugin, we have two different "All Data" options:

* All Data (Heavy): what used to be "All Data"
* All Data (Light): it only sends one value per voxel, the actual data.

> NOTE: It would be valuable to pass this code to Rick (maybe we even did in the past) and potentially make this a default option. That way we won't have to worry about getting this reconfigured every time we run our own experiment. We would also need to check with Vinai, as he is the one who decides when to upgrade AFNI in the scanners.

> NOTE: The modified plugin is in PRJ_rtCAPs/afni_plugin/plug_realtime.c

### 2. Copy processing scripts to EXXXX directory

1. Copy 01_BringROIsToSubjectSpace.sh from PRJ_Dir/Scripts/Realtime_Scanner/01_BringROIsToSubjectSpace.sh to the EXXXX directory in the realtime system. 

> NOTE: This directory only exists after AFNI has started, meaning, you need to wait until the completion of the 3d Localizer to perform this operation.

> NOTE: It may be necessary to change some of the paths

### 3. Copy the Templates to the EXXX directory.

* CAP Templte: This template (Frontiers2013_CAPs.nii) is available in the Others folder in this project and needs to be copied into the EXXXXX dir.

* GM Ribbon Template: This template (MNI_Glasser_HCP_v1.0.nii.gz) is available in the Others folder in this project and needs to be copied into /home/sdc/javiergc/CAPs_Frontiers2013/jan15/ in the realtime system.

> NOTE: If we change 01_BringROIsToSubjectSpace.sh, we could also copy this file to the local EXXXX dir. One plus of this way of doing things is that anyting in the EXXXX gets saved into gold by the realtime system and can be found later.








