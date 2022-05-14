from .defaults import _C as cfg

def parser(config_file):
  cfg.merge_from_file(config_file)
  cfg.freeze()
  return cfg
