import sys
import os
import os.path as osp
import argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rtcap_lib.fMRI import load_fMRI_file, mask_fMRI_img, unmask_fMRI_img
from rtcap_lib.rt_functions import rt_regress_vol
from rtcap_lib.rt_functions import gen_polort_regressors


class iGLM(object):
    def __init__(self,options):
        # Object Initializations
        self.iGLM_prev        = {}
        self.motion_estimates = []
        self.n                =  0               # Counter for number of volumes pre-processed (Start = 1)
        self.t                = -1              # Counter for number of received volumes (Start = 0)
        self.Nv               = None            # Number of voxels in data mask
        self.out_prefix       = options.out_prefix
        self.out_dir          = options.out_dir
        self.iGLM_polort      = options.iGLM_polort
        self.nvols_discard    = options.discard      # Number of volumes to discard from any analysis (won't enter pre-processing)

        # Load Motion Data
        if (options.motion_path == None) or (not osp.exists(options.motion_path)):
            print('++ Motion Information not provided')
            self.iGLM_motion = False
        else:
            print('++ Loading Motion information into memory....')
            self.motion      = np.loadtxt(options.motion_path)
            self.iGLM_motion = True
            print('   Motion Information Dimensions [%s]' % str(self.motion.shape))

        if self.iGLM_motion:
            self.iGLM_num_regressors = self.iGLM_polort + 6
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(self.iGLM_polort)] + ['roll','pitch','yaw','dS','dL','dP']
        else:
            self.iGLM_num_regressors = self.iGLM_polort
            self.nuisance_labels = ['Polort'+str(i) for i in np.arange(self.iGLM_polort)]

        # Load Mask
        if not osp.exists(options.mask_path):
            print('++ ERROR: Mask not found. Program will exit.')
            sys.exit(-1)
        self.mask_img = load_fMRI_file(options.mask_path)
        # Load fMRI Data
        if not osp.exists(options.data_path):
            print('++ ERROR: Data file not founf. Program will exit.')
            sys.exit(-1)
        self.dataImg = load_fMRI_file(options.data_path)
        self.data    = mask_fMRI_img(self.dataImg,self.mask_img)
        [self.Nv,self.Nt] = self.data.shape
        print('++ INFO: Input Masked Data Dimensions [Nv=%d, Nt=%d]' % (self.Nv,self.Nt))
        # Prepare Legendre Polynomials
        if self.iGLM_polort > -1:
            print('++ INFO: Generating Legendre Polynomials of order [%d]' % self.iGLM_polort)
            self.legendre_pols = gen_polort_regressors(self.iGLM_polort,self.Nt)
        else:
            self.legendre_pols = None

    def compute_TR_data(self, motion, extra):
        # Keep a record of motion estimates
        self.motion_estimates.append(motion)
        
        # Update t (it always does)
        self.t = self.t + 1
        # Update n (only if not longer a discard volume)
        if self.t > self.nvols_discard - 1:
            self.n = self.n + 1
        print('[compute_TR_data] - Time point [t=%d, n=%d]' % (self.t, self.n))
        # Read Data from socket
        this_t_data = np.array(extra)
        
        # If first volume, then create empty structures and call it a day (TR)
        if self.t == 0:
            self.Nv            = len(this_t_data)
            print('Number of Voxels Nv=%d' % self.Nv)
            self.Data_FromAFNI = np.array(this_t_data[:,np.newaxis])
            self.Data_iGLM     = np.zeros((self.Nv,1))
            self.iGLM_Coeffs   = np.zeros((self.Nv,self.iGLM_num_regressors,1))
            print(' == [t=%d,n=%d] Init - Data_iGLM.shape     %s' % (self.t, self.n, str(self.Data_iGLM.shape)))
            print(' == [t=%d,n=%d] Init - iGLM_Coeffs.shape   %s' % (self.t, self.n, str(self.iGLM_Coeffs.shape)))
            return 1

        # For any other vol, if still a discard volume
        if self.n == 0:
            self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis],this_t_data[:, np.newaxis]))  # Only keep this one and previous
            self.Data_iGLM     = np.append(self.Data_iGLM,   np.zeros((self.Nv,1)), axis=1)
            self.iGLM_Coeffs   = np.append(self.iGLM_Coeffs, np.zeros((self.Nv,self.iGLM_num_regressors,1)), axis=2)
            return 1
        
        self.Data_FromAFNI = np.hstack((self.Data_FromAFNI[:,-1][:,np.newaxis],this_t_data[:, np.newaxis]))  # Only keep this one and previous
        
        # Do iGLM (if needed)
        # ===================
        if self.iGLM_motion:
            this_t_nuisance = np.concatenate((self.legendre_pols[self.t,:],motion))[:,np.newaxis]
        else:
            this_t_nuisance = (self.legendre_pols[self.t,:])[:,np.newaxis]
        iGLM_data_out, self.iGLM_prev, Bn = rt_regress_vol(self.n, 
                                                           self.Data_FromAFNI[:,-1][:,np.newaxis],
                                                           this_t_nuisance,
                                                           self.iGLM_prev,
                                                           do_operation = True)

        self.Data_iGLM    = np.append(self.Data_iGLM, iGLM_data_out, axis=1)
        self.iGLM_Coeffs  = np.append(self.iGLM_Coeffs, Bn, axis = 2)
        return 1

    def final_steps(self):    
        print('[final_steps] - About to write outputs to disk.')
        out = unmask_fMRI_img(self.Data_iGLM, self.mask_img, osp.join(self.out_dir,self.out_prefix+'.pp_iGLM.nii'))
        for i,lab in enumerate(self.nuisance_labels):
            data = self.iGLM_Coeffs[:,i,:]
            out = unmask_fMRI_img(data, self.mask_img, osp.join(self.out_dir,self.out_prefix+'.pp_iGLM_'+lab+'.nii'))    
        return 1


def process_command_line():
    parser = argparse.ArgumentParser(description="rtCAPs - Spatial Smoothing Step. Based on NIH-neurofeedback software")
    parser_gen = parser.add_argument_group("General Options")
    parser_gen.add_argument("-i", "--input",  action="store", type=str, dest="data_path",   default=None,     help="Path to input dataset.")
    parser_gen.add_argument("-m", "--mask",   action="store", type=str, dest="mask_path",   default=None,     help="Path to mask dataset.")
    parser_gen.add_argument("-o", "--outdir", action="store", type=str, dest="out_dir",    default='./',     help="Path to output directory.")
    parser_gen.add_argument("-p", "--prefix", action="store", type=str, dest="out_prefix",  default='output', help="Output Prefix")
    parser_iglm = parser.add_argument_group("Incremental GLM Options")
    parser_iglm.add_argument("-P","--polort", action="store", type=int,  dest="iGLM_polort", default=2,    help="Order of Legengre Polynomials for iGLM  [default: %(default)s]")
    parser_iglm.add_argument("-M","--motion", action="store", type=str, dest="motion_path", default=None, help="Path to motion file.")    
    parser_iglm.add_argument("--discard",     action="store", type=int, dest="discard",     default=10,    help="Number of volumes to discard (they won't enter the iGLM step)  [default: %(default)s]")
    
    return parser.parse_args()

def main():
    opts = process_command_line()

    if opts.data_path == None:
        print('++ Error: Input Dataset missing.')
        sys.exit(-1)

    if opts.mask_path == None:
        print('++ Error: Mask Dataset missing.')
        sys.exit(-1)

    if opts.motion_path == None:
        print('++ Warning: No motion parameters will be regressed.')

    if opts.iGLM_polort < 0:
        print('++ Warning: No Legendre polynomials will be regressed.')
            
    iGLM_obj = iGLM(opts)

    for t in np.arange(iGLM_obj.Nt):
        if iGLM_obj.iGLM_motion == True:
            motion = iGLM_obj.motion[t,:]
        else:
            motion = None
        print(motion)
        extra  = iGLM_obj.data[:,t]
        iGLM_obj.compute_TR_data(motion, extra)

    # Save results to disk
    iGLM_obj.final_steps()
    

if __name__ == '__main__':
   sys.exit(main())