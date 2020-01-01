import sys
import argparse
import logging
import multiprocessing as mp 
from time import sleep
from psychopy.visual import Window, TextStim

# Setup Logging Infrastructure
log     = logging.getLogger("online_preproc")
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)

screen_size = [512, 288]

# ==================================================
# ======== Functions (for Comm Process)   ==========
# ==================================================
def comm_process(opts, mp_evt_hit, mp_evt_end):
    print(opts)
    sleep(10)
    mp_evt_end.set()
    return

# ===================================================
# =========   Functions (for GUI Process)   =========
# ===================================================
def show_initial_screen(opts):
    #create a window
    if opts.fullscreen:
        ewin = Window(fullscr = opts.fullscreen, allowGUI=False, units='norm')
    else:
        ewin = Window(screen_size, allowGUI=False, units='norm')
        
    #create some stimuli
    text_inst_line01 = TextStim(win=ewin, text='Please fixate on x-hair,',pos=(0.0,0.4))
    text_inst_line02 = TextStim(win=ewin, text='remain awake,',           pos=(0.0,0.28))
    text_inst_line03 = TextStim(win=ewin, text='and let your mind wander.',pos=(0.0,0.16))
    text_inst_chair  = TextStim(win=ewin, text='X', pos=(0,0))

    #plot on the screen
    text_inst_line01.draw()
    text_inst_line02.draw()
    text_inst_line03.draw()
    text_inst_chair.draw()
    ewin.flip()
    return ewin

def processExperimentOptions (self, options=None):
    parser = argparse.ArgumentParser(description="rtCAPs experimental software. Based on NIH-neurofeedback software")
    parser_gen = parser.add_argument_group("General Options")
    parser_gen.add_argument("-d", "--debug", action="store_true", dest="debug",  help="Enable debugging output [%(default)s]", default=False)
    parser_gen.add_argument("-s", "--silent",   action="store_true", dest="silent", help="Minimal text messages [%(default)s]", default=False)
    parser_gen.add_argument("-p", "--tcp_port", help="TCP port for incoming connections [%(default)s]", action="store", default=53214, type=int, dest='tcp_port')
    parser_gen.add_argument("-S", "--show_data", action="store_true",help="display received data in terminal if this option is specified")
    parser_gen.add_argument("--tr",         help="Repetition time [sec]  [default: %(default)s]",                      dest="tr",default=1.0, action="store", type=float)
    parser_gen.add_argument("--ncores",     help="Number of cores to use in the parallel processing part of the code  [default: %(default)s]", dest="n_cores", action="store",type=int, default=10)
    parser_gen.add_argument("--mask",       help="Mask necessary for smoothing operation  [default: %(default)s]",     dest="mask_path", action="store", type=str, default=None, required=True)
    parser_proc   = parser.add_argument_group("Activate/Deactivate Processing Steps")
    parser_proc.add_argument("--no_ema",    help="De-activate EMA Filtering Step [default: %(default)s]", dest="do_EMA",      default=True, action="store_false")
    parser_proc.add_argument("--no_iglm",   help="De-activate iGLM Denoising Step  [default: %(default)s]",             dest="do_iGLM",     default=True, action="store_false")
    parser_proc.add_argument("--no_kalman", help="De-activate Kalman Low-Pass Filter Step  [default: %(default)s]",     dest="do_kalman",   default=True, action="store_false")
    parser_proc.add_argument("--no_smooth", help="De-activate Spatial Smoothing Step  [default: %(default)s]",          dest="do_smooth",   default=True, action="store_false")
    parser_proc.add_argument("--no_snorm",  help="De-activate per-volume spartial Z-Scoring  [default: %(default)s]",   dest="do_snorm",   default=True, action="store_false")
    parser_iglm = parser.add_argument_group("Incremental GLM Options")
    parser_iglm.add_argument("--polort",     help="Order of Legengre Polynomials for iGLM  [default: %(default)s]",     dest="iGLM_polort", default=2, action="store", type=int)
    parser_iglm.add_argument("--no_iglm_motion", help="Do not use 6 motion parameters in iGLM  [default: %(default)s]", dest="iGLM_motion", default=True, action="store_false")
    parser_iglm.add_argument("--nvols",      help="Number of expected volumes (for legendre pols only)  [default: %(default)s]", dest="nvols",default=500, action="store", type=int, required=True)
    parser_iglm.add_argument("--discard",    help="Number of volumes to discard (they won't enter the iGLM step)  [default: %(default)s]",  default=10, dest="discard", action="store", type=int)
    parser_smo = parser.add_argument_group("Smoothing Options")
    parser_smo.add_argument("--fwhm",      help="FWHM for Spatial Smoothing in [mm]  [default: %(default)s]",          dest="FWHM",        default=4.0, action="store", type=float)
    parser_save   = parser.add_argument_group("Saving Options")
    parser_save.add_argument("--out_dir",     help="Output directory  [default: %(default)s]",                           dest="out_dir",    action="store", type=str, default="./")
    parser_save.add_argument("--out_prefix",  help="Prefix for outputs  [default: %(default)s]",                         dest="out_prefix", action="store", type=str, default="online_preproc")
    parser_save.add_argument("--save_ema",    help="Save 4D EMA dataset  [default: %(default)s]",     dest="save_ema",   default=False, action="store_true")
    parser_save.add_argument("--save_kalman", help="Save 4D Smooth dataset  [default: %(default)s]",     dest="save_kalman",   default=False, action="store_true")
    parser_save.add_argument("--save_smooth", help="Save 4D Smooth dataset  [default: %(default)s]",     dest="save_smooth",   default=False, action="store_true")
    parser_save.add_argument("--save_iglm  ", help="Save 4D iGLM datasets  [default: %(default)s]",     dest="save_iglm",   default=False, action="store_true")
    parser_save.add_argument("--save_orig"  , help="Save 4D with incoming data  [default: %(default)s]", dest="save_orig", default=False, action="store_true")
    parser_save.add_argument("--save_all"  ,  help="Save 4D with incoming data  [default: %(default)s]", dest="save_all", default=False, action="store_true")
    parser_exp = parser.add_argument_group('Experiment/GUI Options')
    parser_exp.add_argument("-e","--exp_type", help="Type of Experimental Run [%(default)s]",      type=str, required=True,  choices=['preproc','esam'], default='preproc')
    parser_exp.add_argument("--no_proc_chair", help="Hide crosshair during preprocessing run [%(default)s]", default=False,  action="store_true", dest='no_proc_chair')
    parser_exp.add_argument("--fscreen", help="Use full screen for Experiment [%(default)s]", default=False, action="store_true", dest="fullscreen")
    parser_exp.add_argument("--screen", help="Monitor to use [%(default)s]", default=1, action="store", dest="screen",type=int)
    parser_dec = parser.add_argument_group('SVR/Decoding Options')
    parser_dec.add_argument("--svr_start",  help="Volume when decoding should start. When we think iGLM is sufficient_stable [%(default)s]", default=100, dest="dec_start_vol", action="store", type=int)
    parser_dec.add_argument("--svr_path",   help="Path to pre-trained SVR models [%(default)s]", dest="svr_path", action="store", type=str, default=None)
    parser_dec.add_argument("--svr_zth",    help="Z-score threshold for deciding hits [%(default)s]", dest="hit_zth", action="store", type=float, default=2.0)
    parser_dec.add_argument("--svr_vhit",   help="Number of consecutive vols over threshold required for a hit [%(default)s]", dest="hit_v4hit", action="store", type=int, default=2)
    parser_dec.add_argument("--svr_win_activate", help="Activate windowing of individual volumes prior to hit estimation [%(default)s]", dest="hit_dowin", action="store_true", default=False)
    parser_dec.add_argument("--svr_win_wl", help='Number of volumes for SVR windowing step [%(default)s]', dest='hit_wl', default=4, type=int, action='store')
    parser_dec.add_argument("--svr_mot_activate", help="Consider a hit if excessive motion [%(default)s]", dest="hit_domot", action="store_true", default=False )
    parser_dec.add_argument("--svr_mot_th", help="Framewise Displacement Treshold for motion [%(default)s]",  action="store", type=float, dest="svr_mot_th", default=1.2)
    parser_dec.add_argument("--svr_hit_mehod", help="Method for deciding hits [%(default)s]", type=str, choices=["method01"], default="method01", action="store", dest="hit_method")


    return parser.parse_args(options)

def main():
    # 1) Read Input Parameters: port, fullscreen, etc..
    log.info('1) Reading input parameters...')
    opts = processExperimentOptions(sys.argv)
    log.debug('User Options: %s' % str(opts))

    # 2) Create Multi-processing infra-structure
    # ------------------------------------------
    mp_evt_hit = mp.Event()
    mp_evt_end = mp.Event()

    mp_prc_comm = mp.Process(target=comm_process, args=(opts, mp_evt_hit, mp_evt_end))
    mp_prc_comm.start()

    # 3) Start GUI
    # ------------
    ewin = show_initial_screen(opts)

    # 4) Wait for things to happen
    # ----------------------------
    while not mp_evt_end.is_set():
        sleep(0.1)
        if mp_evt_hit.is_set():
            do_questionaire()
        print('waiting')

if __name__ == '__main__':
   sys.exit(main())