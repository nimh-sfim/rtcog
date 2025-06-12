import os.path as osp

def file_exists(path):
      if not osp.isfile(path):
         raise FileNotFoundError(f"File not found: {path}")
      return path

