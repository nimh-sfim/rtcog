import os.path as osp
import argparse
import yaml

class Options:
     def __init__(self, config):
         """Class for holding experiment options. Allows users to build from yaml file or cli args."""
     #     self.config = config
         self.__dict__.update(config)

     @staticmethod    
     def file_exists(path):
          if not osp.isfile(path):
               raise FileNotFoundError(f"File not found: {path}")
          return path

     @staticmethod
     def load_yaml(path):
          with open(path, 'r') as f:
             config = yaml.safe_load(f)
          return config

     def parse_cli_args(self, argv=None):
          """Load config file if given, and override with any other options provided"""
          config = {}
          pre_parser = argparse.ArgumentParser(add_help=False)
          pre_parser.add_argument("-c", "--config", dest="config_path", help="yaml file containing experiment options")
          pre_args, remaining_args = pre_parser.parse_known_args(argv)

          if pre_args.config_path:
               if not osp.exists(pre_args.config_path):
                    raise FileNotFoundError(f"Config file {pre_args.config_path} not found")
               # Load options from yaml file and return as Namespace
               config = self.load_yaml(pre_args.config_path)

          # Override yaml file with any new options
          parser = argparse.ArgumentParser(
          description="rtCAPs experimental software. Based on NIH-neurofeedback software. "
                         "You can optionally provide a yaml config file via --config to set all options."
          )

          parser_gen = parser.add_argument_group("General Options")
          parser_gen.add_argument("-d", "--debug", action="store_true", dest="debug",  help="Enable debugging output [%(default)s]", default=False)
          parser_gen.add_argument("-s", "--silent",   action="store_true", dest="silent", help="Minimal text messages [%(default)s]", default=False)
          parser_gen.add_argument("-p", "--tcp_port", help="TCP port for incoming connections [%(default)s]", action="store", default=53214, type=int, dest='tcp_port')
          parser_gen.add_argument("-S", "--show_data", action="store_true",help="display received data in terminal if this option is specified")
          parser_gen.add_argument("--tr",         help="Repetition time [sec]  [default: %(default)s]",                      dest="tr",default=1.0, action="store", type=float)
          parser_gen.add_argument("--ncores",     help="Number of cores to use in the parallel processing part of the code  [default: %(default)s]", dest="n_cores", action="store",type=int, default=10)
          parser_gen.add_argument("--mask",       help="Mask necessary for smoothing operation  [default: %(default)s]",     dest="mask_path", action="store", type=Options.file_exists, default=None, required=True)
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
          parser_exp.add_argument("-e","--exp_type", help="Type of Experimental Run [%(default)s]",      type=str, required=True,  choices=['preproc','esam', 'esam_test'], default='preproc')
          parser_exp.add_argument("--no_gui", help="Do not open psychopy window. Only applies to pre-processing experiment type [%(default)s]", default=False, action="store_true", dest='no_gui')
          parser_exp.add_argument("--no_proc_chair", help="Hide crosshair during preprocessing run [%(default)s]", default=False,  action="store_true", dest='no_proc_chair')
          parser_exp.add_argument("--fscreen", help="Use full screen for Experiment [%(default)s]", default=False, action="store_true", dest="fullscreen")
          parser_exp.add_argument("--screen", help="Monitor to use [%(default)s]", default=1, action="store", dest="screen",type=int)
          parser_dec = parser.add_argument_group('SVR/Decoding Options')
          parser_dec.add_argument("--svr_start",  help="Volume when decoding should start. When we think iGLM is sufficient_stable [%(default)s]", default=100, dest="dec_start_vol", action="store", type=int)
          parser_dec.add_argument("--svr_path",   help="Path to pre-trained SVR models [%(default)s]", dest="svr_path", action="store", type=Options.file_exists, default=None)
          parser_dec.add_argument("--svr_zth",    help="Z-score threshold for deciding hits [%(default)s]", dest="hit_zth", action="store", type=float, default=1.75)
          parser_dec.add_argument("--svr_consec_vols",   help="Number of consecutive vols over threshold required for a hit [%(default)s]", dest="nconsec_vols", action="store", type=int, default=2)
          parser_dec.add_argument("--svr_win_activate", help="Activate windowing of individual volumes prior to hit estimation [%(default)s]", dest="hit_dowin", action="store_true", default=False)
          parser_dec.add_argument("--svr_win_wl", help='Number of volumes for SVR windowing step [%(default)s]', dest='hit_wl', default=4, type=int, action='store')
          parser_dec.add_argument("--svr_mot_activate", help="Consider a hit if excessive motion [%(default)s]", dest="hit_domot", action="store_true", default=False )
          parser_dec.add_argument("--svr_mot_th", help="Framewise Displacement Treshold for motion [%(default)s]",  action="store", type=float, dest="svr_mot_th", default=1.2)
          parser_dec.add_argument("--svr_hit_mehod", help="Method for deciding hits [%(default)s]", type=str, choices=["method01"], default="method01", action="store", dest="hit_method")
          parser_dec.add_argument("--svr_vols_noqa", help="Min. number of volumes to wait since end of last QA before declaing a new hit. [%(default)s]", type=int, dest='vols_noqa', default=45, action="store")
          parser_dec.add_argument("--q_path", help="The path to the questions json file containing the question stimuli. If not a full path, it will default to look in RESOURCES_DIR", type=str, dest='q_path', default="questions_v1", action="store")
          parser_dec = parser.add_argument_group('Testing Options')
          parser_dec.add_argument("--snapshot",  help="Run snapshot test", default=False, dest="snapshot", action="store_true")

          cli_args = parser.parse_args(remaining_args)

          cli_dict = vars(cli_args)
          config.update({k: v for k, v in cli_dict.items() if v is not None})
          
          return config
          
     @classmethod
     def from_yaml(cls, path):
          config = cls.load_yaml(path)
          return cls(config)
     
     @classmethod
     def from_cli(cls, argv=None):
          config = cls.parse_cli_args(argv)
          return cls(config) 
     
     def __repr__(self):
          return f"Options: {self.__dict__}"

     def save_config(self):
          print(self.out_dir)
          out_path = osp.join(self.out_dir, f'{self.out_prefix}_Options.yaml')
          with open(out_path, 'w') as file:
               yaml.safe_dump(self.__dict__, sort_keys=False)

        
