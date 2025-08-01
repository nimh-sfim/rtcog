import os.path as osp
import sys
import argparse
import yaml

from rtfmri.utils.core import file_exists

class Options:
     """
     Configuration object for the real-time fMRI pipeline.

     Loads configuration from a YAML file and/or command-line arguments,
     and exposes them as attributes for easy access throughout the experiment code.

     Attributes
     ----------
     (Dynamically assigned)
          All configuration keys are stored as object attributes.
     """
     def __init__(self, config):
          """
          Initialize the Options object with a given config dictionary.

          Parameters
          ----------
          config : dict
               Dictionary containing configuration keys and values.
          """
          self.__dict__.update(config)

     @staticmethod
     def load_yaml(path):
          """
          Load configuration from a YAML file.

          Parameters
          ----------
          path : str
               Path to the YAML configuration file.

          Returns
          -------
          dict
               Parsed YAML content as a dictionary.
          """
          with open(path, 'r') as file:
             config = yaml.safe_load(file)
          return config
     
     @classmethod
     def parse_cli_args(cls, argv=None):
          """
          Parse command-line arguments to load and optionally override config options.

          Parameters
          ----------
          argv : list or None
               Command-line arguments.

          Returns
          -------
          dict
               Merged configuration from YAML and CLI overrides.

          Raises
          ------
          FileNotFoundError
               If the specified YAML config file cannot be found.
          SystemExit
               If required arguments are missing.
          """
          config = {}
          pre_parser = argparse.ArgumentParser(add_help=False)
          pre_parser.add_argument("-c", "--config", dest="config_path", help="yaml file containing experiment options", required=True)
          pre_args, remaining_args = pre_parser.parse_known_args(argv)

          if pre_args.config_path:
               if not osp.exists(pre_args.config_path):
                    raise FileNotFoundError(f"Config file {pre_args.config_path} not found")
               # Load options from yaml file and return as Namespace
               config = cls.load_yaml(pre_args.config_path)
          
          # Override yaml file with any new options
          parser = argparse.ArgumentParser(
          description = "rtCAPs experimental software. Based on NIH-neurofeedback software. "
                        "Provide a yaml config file via --config/-c to set all options. "
                        "You can override some options via CLI."
          )

          parser_gen = parser.add_argument_group("General Options")
          parser_gen.add_argument("-d", "--debug", action="store_true", dest="debug", help="Enable debugging output", default=None)
          parser_gen.add_argument("-s", "--silent",   action="store_true", dest="silent", help="Minimal text messages", default=None)
          parser_gen.add_argument("-p", "--tcp_port", help="TCP port for incoming connections", action="store", type=int, dest='tcp_port')
          parser_gen.add_argument("-S", "--show_data", action="store_true",help="display received data in terminal if this option is specified", default=None)
          parser_gen.add_argument("--tr", help="Repetition time [sec]", dest="tr", action="store", type=float)
          parser_gen.add_argument("--ncores", help="Number of cores to use in the parallel processing part of the code", dest="n_cores", action="store",type=int)
          parser_gen.add_argument("-m","--mask", help="Path to mask", dest="mask_path", action="store", type=file_exists)
          
          parser_iglm = parser.add_argument_group("Incremental GLM Options")
          parser_iglm.add_argument("--polort", help="Order of Legengre Polynomials for iGLM",dest="iGLM_polort", action="store", type=int)
          parser_iglm.add_argument("--no_iglm_motion", help="Do not use 6 motion parameters in iGLM", dest="iGLM_motion", action="store_false", default=None)
          parser_iglm.add_argument("--nvols", help="Number of expected volumes (for legendre pols only)", dest="nvols", action="store", type=int)
          parser_iglm.add_argument("--discard", help="Number of volumes to discard (they won't enter the iGLM step)", dest="discard", action="store", type=int)

          parser_smo = parser.add_argument_group("Smoothing Options")
          parser_smo.add_argument("--fwhm", help="FWHM for Spatial Smoothing in [mm]", dest="FWHM", action="store", type=float)

          parser_save = parser.add_argument_group("Saving Options")
          parser_save.add_argument("--out_dir", help="Output directory", dest="out_dir", action="store", type=str)
          parser_save.add_argument("--out_prefix", help="Prefix for outputs", dest="out_prefix", action="store", type=str)
          parser_save.add_argument("--auto_save", help="Automatically save all outputs even if error is encountered during processing.", dest="auto_save", action="store_true", default=None)
          
          parser_exp = parser.add_argument_group('Experiment/GUI Options')
          parser_exp.add_argument("-e","--exp_type", help="Type of Experimental Run", type=str, choices=['preproc','esam', 'esam_test'])
          parser_exp.add_argument("--no_gui", help="Do not open psychopy window. Only applies to pre-processing experiment type", action="store_true", dest='no_gui', default=None)
          parser_exp.add_argument("--no_proc_chair", help="Hide crosshair during preprocessing run", action="store_true", dest='no_proc_chair', default=None)
          parser_exp.add_argument("--fscreen", help="Use full screen for Experiment", action="store_true", dest="fullscreen", default=None)
          parser_exp.add_argument("--screen", help="Monitor to use", action="store", dest="screen",type=int)
          parser_exp.add_argument("--q_path", help="The path to the questions json file containing the question stimuli. If not a full path, it will look for the file in RESOURCES_DIR", type=str, dest='q_path', action="store")

          parser_dec = parser.add_argument_group('Matching Options')
          parser_dec.add_argument("--match_path", help="Path to inputs required for matching method", dest="match_path", action="store", type=file_exists, default=None)
          parser_dec.add_argument("--hit_thr", help="Threshold for deciding hits [%(default)s]", dest="hit_thr", action="store", type=float, default=None)

          parser_dec = parser.add_argument_group('Testing Options')
          parser_dec.add_argument("--snapshot", help="Run snapshot test", dest="snapshot", action="store_true", default=None)
          parser_dec.add_argument("--latency", help="Run latency test", dest="test_latency", action="store_true", default=None)

          cli_args = parser.parse_args(remaining_args)

          # Only override yaml config with CLI args that were explicitly passed
          for k, v in vars(cli_args).items():
               if v is not None:
                    config[k] = v
          
          if 'exp_type' not in config:
               print(f"++ ERROR: Please specify experiment type with -e/-exp_type.")
               sys.exit(-1)

          required_args = ['mask_path', 'nvols', 'out_dir', 'out_prefix']
          if config['exp_type'] == 'esam':
               required_args.extend(['match_path', 'hit_thr'])

          missing = []
          for arg in required_args:
               if not arg in config or config[arg] is None:
                    missing.append(f'--{arg}') 
          if missing:
               print(f"++ ERROR: The following arguments are required: {', '.join(missing)}")
               sys.exit(-1)
                    
          return config


     @classmethod
     def from_yaml(cls, path):
          """
          Create an Options object from a YAML file.

          Useful for testing purposes.

          Parameters
          ----------
          path : str
               Path to the YAML configuration file.

          Returns
          -------
          Options
               An instance populated with the file contents.
          """
          config = cls.load_yaml(path)
          return cls(config)
     
     @classmethod
     def from_cli(cls, argv=None):
          """
          Create an Options object from CLI arguments (with YAML config as base).

          Parameters
          ----------
          argv : list or None
               Command-line arguments.

          Returns
          -------
          Options
               An instance with merged YAML and CLI arguments.
          """
          config = cls.parse_cli_args(argv)
          return cls(config) 
     
     def __str__(self):
          return f"Options:\n{yaml.dump(self.__dict__, sort_keys=False)}"

     def save_config(self):
          """
          Save the current configuration to a YAML file in the output directory.
          The filename will include the out_prefix (e.g., `Run01_Options.yaml`).
          """
          out_path = osp.join(self.out_dir, f'{self.out_prefix}_Options.yaml')
          with open(out_path, 'w') as file:
               yaml.safe_dump(self.__dict__, file, sort_keys=False)
          print(f"++ Options saved to {out_path}")

        
