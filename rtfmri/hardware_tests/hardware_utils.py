import argparse

def get_opts(options=None):
    parser = argparse.ArgumentParser(
        description="Make sure psychopy is detecting keystrokes"
    )

    parser_save = parser.add_argument_group("Saving Options")
    parser_save.add_argument("--out_dir", help="Output directory  [default: %(default)s]", dest="out_dir", action="store", type=str, default="./")
    parser_save.add_argument("--out_prefix",  help="Prefix for outputs", dest="out_prefix", action="store", type=str)

    return parser.parse_args(options)