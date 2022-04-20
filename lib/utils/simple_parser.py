"""
adapted from  Detectron config file (https://github.com/facebookresearch/Detectron)
"""

import os
import yaml
import random
import copy
from ast import literal_eval
from fractions import Fraction
import logging

path = os.path.dirname(__file__)

class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        elif name.startswith('__'):
            raise AttributeError(name)
        else:
            self[name] = AttrDict()
            return self[name]

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    def __str__(self):
        return yaml.dump(self.strip(), default_flow_style=False)

    def merge(self, other):
        if not isinstance(other, AttrDict):
            other = AttrDict.cast(other)

        for k, v in other.items():
            v = copy.deepcopy(v)
            if k not in self or not isinstance(v, dict):
                self[k] = v
                continue
            AttrDict.__dict__['merge'](self[k], v)

    def strip(self):
        if not isinstance(self, dict):
            if isinstance(self, list) or isinstance(self, tuple):
                self = str(tuple(self))
            return self
        return {k: AttrDict.__dict__['strip'](v) for k, v in self.items()}

    #@classmethod
    #def cast(C, d):
    #    if not isinstance(d, dict):
    #        return d
    #    return C({k: C.cast(v) for k, v in d.items()})

    #https://stackoverflow.com/questions/13183501/staticmethod-and-recursion
    @staticmethod
    def cast(d):
        if not isinstance(d, dict):
            return d
        return AttrDict({k: AttrDict.cast(v) for k, v in d.items()})


def parse(d):
    # parse string as tuple, list or fraction
    if not isinstance(d, dict):
        if isinstance(d, str):
            try:
                d = literal_eval(d)
            except:
                try:
                    d = float(Fraction(d))
                except:
                    pass
        return d
    return AttrDict({k: parse(v) for k, v in d.items()})

def load(fname):
    with open(fname, 'r', encoding='UTF-8') as f:
        ret = parse(yaml.load(f, Loader=yaml.FullLoader))
    return ret

class Parser(AttrDict):
    def __init__(self, cfg_name='', ):
        if cfg_name:
            self.add_cfg(cfg_name)
            # setup(self, log)

    def add_args(self, args):
        self.merge(vars(args))
        return self

    def add_cfg(self, cfg, args=None, update=False):
        if os.path.isfile(cfg):
            fname = cfg
            cfg = os.path.splitext(os.path.basename(cfg))[0]
        else:
            assert False

        self.merge(load(fname))
        self['name'] = cfg

        if args is not None:
            self.add_args(args)
        return self



