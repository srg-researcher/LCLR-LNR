import yaml

global cfg
if 'cfg' not in globals():
    with open('/nesi/project/uoo03832/LCLR-LNR-FL/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)