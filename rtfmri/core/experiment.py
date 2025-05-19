import logging
import sys
import os.path as osp
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas

from core.preproc import Pipeline
from utils.log import get_logger, set_logger
from paths import CAP_labels

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
sys.path.append('..')

from utils.fMRI import load_fMRI_file, unmask_fMRI_img


class Experiment:
    def __init__(self, options, mp_evt_hit, mp_evt_end, mp_evt_qa_end):
        self.log = set_logger(options.debug, options.silent)

        self.mp_evt_hit = mp_evt_hit           # Signals a CAP hit
        self.mp_evt_end = mp_evt_end           # Signals the end of the experiment
        self.mp_evt_qa_end = mp_evt_qa_end     # Signals the end of a QA set

        self.ewin          = None
        self.exp_type      = options.exp_type
        self.no_proc_chair = options.no_proc_chair
        self.screen_size   = [512, 288]
        self.fullscreen    = options.fullscreen
        self.screen        = options.screen

        self.n             = 0               # Counter for number of volumes pre-processed (Start = 1)
        self.t             = -1              # Counter for number of received volumes (Start = 0
        self.lastQA_endTR  = 0
        self.vols_noqa     = options.vols_noqa
        self.Nv            = None            # Number of voxels in data mask
        self.Nt            = options.nvols   # Number acquisitions
        self.TR            = options.tr      # TR [seconds]

        self.nvols_discard = options.discard      # Number of volumes to discard from any analysis (won't enter pre-processing)

        if options.mask_path is None:
            self.log.warning('  Experiment_init_ - No mask was provided!')
            self.mask_img = None
        else:
            self.mask_img  = load_fMRI_file(options.mask_path)
            self.mask_Nv = int(np.sum(self.mask_img.get_fdata()))
            self.log.debug('  Experiment_init_ - Number of Voxels in user-provided mask: %d' % self.mask_Nv)

        self.motion_estimates = []
        
        self.pipe = Pipeline(options, self.Nt, self.mask_Nv, self.mask_img, self.exp_type)        

    def compute_TR_data(self, motion, extra):
        self.t += 1

        # Keep a record of motion estimates
        motion = [i[self.t] for i in motion]
        self.motion_estimates.append(motion)

        if len(motion) != 6:
            self.log.error('Motion not read in correctly.')
            self.log.error(f'Expected length: 6 | Actual length: {len(motion)}')
            sys.exit(-1)
        
        this_t_data = np.array([e[self.t] for e in extra])
        
        self.Nv = len(this_t_data)

        if self.t > 0:
            if len(this_t_data) != self.Nv:
                self.log.error(f'Extra data not read in correctly.')
                self.log.error(f'Expected length: {self.Nv} | Actual length: {len(this_t_data)}')
                sys.exit(-1)
        
        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n += 1
        
        self.pipe.process(self.t, self.n, motion, this_t_data)
        
        del extra # Save resources
        
        return 1
        
    def final_steps(self):
        # Write out motion
        self.motion_estimates = [item for sublist in self.motion_estimates for item in sublist]
        self.log.info('self.motion_estimates length is %d' % len(self.motion_estimates))
        self.motion_estimates = np.reshape(self.motion_estimates,newshape=(int(len(self.motion_estimates)/6),6))
        np.savetxt(osp.join(self.out_dir,self.out_prefix+'.Motion.1D'), 
                   self.motion_estimates,
                   delimiter="\t")
        self.log.info('Motion estimates saved to disk: [%s]' % osp.join(self.out_dir,self.out_prefix+'.Motion.1D'))

        if self.mask_img is None:
            self.log.warning(' final_steps = No additional outputs generated due to lack of mask.')
            return 1
        
        self.log.debug(' final_steps - About to write outputs to disk.')
        out_vars   = [self.Data_norm]
        out_labels = ['.pp_Zscore.nii']
        if self.do_EMA and self.save_ema:
            out_vars.append(self.Data_EMA)
            out_labels.append('.pp_EMA.nii')
        if self.do_iGLM and self.save_iGLM:
            out_vars.append(self.Data_iGLM)
            out_labels.append('.pp_iGLM.nii')
        if self.do_kalman and self.save_kalman:
            out_vars.append(self.Data_kalman)
            out_labels.append('.pp_LPfilter.nii')
        if self.do_smooth and self.save_smooth:
            out_vars.append(self.Data_smooth)
            out_labels.append('.pp_Smooth.nii')
        if self.save_orig:
            out_vars.append(self.Data_FromAFNI)
            out_labels.append('.orig.nii')
        for variable, file_suffix in zip(out_vars, out_labels):
            unmask_fMRI_img(variable, self.mask_img, osp.join(self.out_dir,self.out_prefix+file_suffix))

        if self.do_iGLM and self.save_iGLM:
            for i,lab in enumerate(self.nuisance_labels):
                data = self.iGLM_Coeffs[:,i,:]
                unmask_fMRI_img(data, self.mask_img, osp.join(self.out_dir,self.out_prefix+'.pp_iGLM_'+lab+'.nii'))    

        if self.exp_type == "esam" or self.exp_type == "esam_test":
            svrscores_path = osp.join(self.out_dir,self.out_prefix+'.svrscores')
            np.save(svrscores_path,self.svrscores)
            self.log.info('Saved svrscores to %s' % svrscores_path)
            hits_path = osp.join(self.out_dir,self.out_prefix+'.hits')
            np.save(hits_path, self.hits)
            self.log.info('Saved hits info to %s' % hits_path)
        self.log.info(' - final_steps - Setting end of experiment event (mp_evt_end)')

        # Write out the dynamic report for this run
        if self.exp_type == "esam" or self.exp_type == "esam_test":
            SVR_Scores_DF       = pd.DataFrame(self.svrscores.T, columns=CAP_labels)
            SVR_Scores_DF['TR'] = SVR_Scores_DF.index
            SVRscores_curve     = SVR_Scores_DF.hvplot(legend='top', label='SVR Scores', x='TR').opts(width=1500)
            Threshold_line      = hv.HLine(self.hit_zth).opts(color='black', line_dash='dashed', line_width=1)
            Hits_ToPlot         = self.hits.T * self.svrscores.T
            Hits_ToPlot[Hits_ToPlot==0.0] = None
            Hits_DF             = pd.DataFrame(Hits_ToPlot, columns=CAP_labels)
            Hits_DF['TR']       = Hits_DF.index
            Hits_Marks          = Hits_DF.hvplot(legend='top', label='SVR Scores', 
                                             x='TR', kind='scatter', marker='circle', 
                                             alpha=0.5, s=100).opts(width=1500)
            qa_boxes = []
            for (on,off) in zip(self.qa_onsets,self.qa_offsets):
                qa_boxes.append(hv.Box(x=on+((off-on)/2),y=0,spec=(off-on,10)))
            QA_periods = hv.Polygons(qa_boxes).opts(alpha=.2, color='blue', line_color=None)
            wait_boxes = []
            for off in self.qa_offsets:
                wait_boxes.append(hv.Box(x=off+(self.vols_noqa/2),y=0,spec=(self.vols_noqa,10)))
            WAIT_periods = hv.Polygons(wait_boxes).opts(alpha=.2, color='cyan', line_color=None)
            plot_layout = (SVRscores_curve * Threshold_line * Hits_Marks * QA_periods * WAIT_periods).opts(title='Experimental Run Results:'+self.out_prefix)
            renderer    = hv.renderer('bokeh')
            renderer.save(plot_layout, self.outhtml)
            self.log.info(' - final_steps - Dynamic Report written to disk: [%s.html]' % self.outhtml)
            self.log.info(' - final_steps - qa_onsets:  %s' % str(self.qa_onsets))
            self.log.info(' - final_steps - qa_offsets: %s' % str(self.qa_offsets))

        # Write out the maps associated with the hits
        if self.exp_type == "esam" or self.exp_type == "esam_test":
            Hits_DF = pd.DataFrame(self.hits.T, columns=CAP_labels)
            for cap in CAP_labels:
                thisCAP_hits = Hits_DF[cap].sum()
                if thisCAP_hits > 0: # There were hits for this particular cap
                    hit_ID = 1
                    for vol in Hits_DF[Hits_DF[cap]==True].index:
                        if self.hit_dowin == True:
                            thisCAP_Vols = vol-np.arange(self.hit_wl+self.nconsec_vols-1)
                        else:
                            thisCAP_Vols = vol-np.arange(self.nconsec_vols)
                        out_file = osp.join(self.out_dir, self.out_prefix + '.Hit_'+cap+'_'+str(hit_ID).zfill(2)+'.nii')
                        self.log.info(' - final_steps - [%s-%d]. Contributing Vols: %s | File: %s' % (cap, hit_ID,str(thisCAP_Vols), out_file ))
                        self.log.debug(' - final_steps - self.Data_norm.shape %s' % str(self.Data_norm.shape))
                        self.log.debug(' - final_steps - self.Data_norm[:,thisCAP_Vols].shape %s' % str(self.Data_norm[:,thisCAP_Vols].shape))
                        thisCAP_InMask  = self.Data_norm[:,thisCAP_Vols].mean(axis=1)
                        self.log.debug(' - final_steps - thisCAP_InMask.shape %s' % str(thisCAP_InMask.shape))
                        unmask_fMRI_img(thisCAP_InMask, self.mask_img, out_file)
                        hit_ID = hit_ID + 1

            # Write out QA_Onsers and QA_Offsets
            with open(self.qa_onsets_path,'w') as file:
                for onset in self.qa_onsets:
                    file.write("%i\n" % onset)
            with open(self.qa_offsets_path,'w') as file:
                for offset in self.qa_offsets:
                    file.write("%i\n" % offset)        
        
        # If running snapshot test, save the variable states
        if self.snapshot:
            var_dict = {
                'Data_norm': self.Data_norm,
                'Data_EMA': self.Data_EMA,
                'Data_iGLM': self.Data_iGLM,
                'Data_smooth': self.Data_smooth,
                # 'Data_kalman': self.Data_kalman,
                'Data_FromAFNI': self.Data_FromAFNI
            }

            np.savez(osp.join(self.out_dir, f'{self.out_prefix}_snapshots.npz'), **var_dict) 

        # Inform other threads that this is comming to an end
        self.mp_evt_end.set()


