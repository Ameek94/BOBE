# Initialization methods for the samplers and optimizers using inputs from yaml
#  Default settings in default.yaml

# needs some cleanup

import os
from typing import Any, Optional
import yaml
from yaml import load, dump
import re
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# for scientific notation - from https://stackoverflow.com/a/30462009, see also cobaya yaml.py
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


default_file = os.path.join(os.path.dirname(__file__),'./default.yaml')

class input_settings:

    settings: dict[str,Any]

    def __init__(self
                 ,set_from_file: bool
                 ,file: Optional[str]
                 ,input_dict: Optional[dict[str,Any]] = {}) -> None:
        
        with open(default_file,'r') as f:
            self.defaults = yaml.load(f,Loader=loader)
        
        if set_from_file:
            try:
                assert file is not None
                with open(file,'r') as f:
                    self.settings = yaml.load(f,Loader=loader) 
            except FileNotFoundError:
                self.settings = self.defaults
                print("Run settings not found, reverting to defaults")
        else:
            self.settings = input_dict # type: ignore
            
        self.set_gp_settings()
        self.set_acq_settings()
        self.set_ns_settings()
        self.set_optimizer_settings()
        self.set_bo_settings()

        self.settings = {}
        self.settings["BO"] = self.bo_settings
        self.settings["NS"] = self.ns_settings
        self.settings["GP"] = self.gp_settings
        self.settings["ACQ"] = self.acq_settings
        self.settings["optimizer"] = self.optimzer_settings

    def set_gp_settings(self):
        method = self.set_method("GP")
        self.gp_settings = {'method': method}
        self.gp_settings[method] = self.set_from_file("GP",method)

    def set_ns_settings(self):
        method = self.set_method("NS")
        self.ns_settings = {'method': method}
        self.ns_settings[method] = self.set_from_file("NS",method)

    def set_acq_settings(self):
        method = self.set_method("ACQ")
        self.acq_settings = {'method': method}
        self.acq_settings[method] = self.set_from_file("ACQ",method)
    
    def set_optimizer_settings(self):
        method = self.set_method("optimizer")
        self.optimzer_settings = {'method': method}
        self.optimzer_settings[method] = self.set_from_file("optimizer",method)

    def set_bo_settings(self):
        method = self.set_method("BO")
        self.bo_settings = {'method': method}
        self.bo_settings[method] = self.set_from_file("BO",method)
    
    def set_method(self
                   ,module: str):
        if module in self.defaults.keys():
            try:
                if 'method' in self.settings[module].keys():
                    method = self.settings[module]['method']
                else:
                    method = self.defaults[module]['method']
            except:
                method = self.defaults[module]['method']
        else:
            method = self.defaults[module]['method']
        return method

    def set_from_file(self
                      ,module:str
                      ,method:str):
        settings = {}
        if module in self.settings.keys():
            for key in self.defaults[module][method]:
                try:
                    if key in self.settings[module][method]:
                        val = self.settings[module][method][key]
                    else:
                        val = self.defaults[module][method][key]
                except:
                    val = self.defaults[module][method][key]
                settings[key] = val
        else:
            settings = self.defaults[module][method]
        return settings