import sys
import os
import argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rtcap_lib.fMRI import load_fMRI_file, mask_fMRI_img, unmask_fMRI_img
from rtcap_lib.rt_functions import rt_snorm_vol

def process_command_line():
    parser = argparse.ArgumentParser(description="rtCAPs - Spatial Normalization Step. Based on NIH-neurofeedback software")
    parser_gen = parser.add_argument_group("General Options")
    parser_gen.add_argument("-i", "--input",  action="store", type=str, dest="data_path", help="Path to input dataset.",  default=None)
    parser_gen.add_argument("-m", "--mask",   action="store", type=str, dest="mask_path", help="Path to mask dataset.",   default=None)
    parser_gen.add_argument("-o", "--output", action="store", type=str, dest="out_path",  help="Path to output dataset.", default=None)
    return parser.parse_args()
    
def main():
    opts = process_command_line()

    if opts.data_path == None:
        print('++ Error: Input Dataset missing.')
        sys.exit(-1)

    if opts.mask_path == None:
        print('++ Error: Mask Dataset missing.')
        sys.exit(-1)

    if opts.out_path == None:
        print('++ Error: Output Dataset missing.')
        sys.exit(-1)

    # Load Data
    DataImg = load_fMRI_file(opts.data_path)
    MaskImg = load_fMRI_file(opts.mask_path)

    Data    = mask_fMRI_img(DataImg,MaskImg)
    [Nv,Nt] = Data.shape
    print('Input Masked Data Dimensions [Nv=%d,Nt=%d]' % (Nv,Nt))

    # Perform Processing Step
    outData = np.zeros((Nv,Nt))
    for t in np.arange(Nt):
        aux          = Data[:,t]
        zaux         = rt_snorm_vol(aux)
        outData[:,t] = np.squeeze(zaux)

    # Save results to disk
    final = unmask_fMRI_img(outData,MaskImg,opts.out_path)
    

if __name__ == '__main__':
   sys.exit(main())
