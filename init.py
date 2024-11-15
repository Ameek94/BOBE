# Initialization methods for the samplers and optimizers using inputs from yaml
#  Default settings in default.yaml

import yaml
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class input_settings:

    def __init__(self,file) -> None:
        
        with open('./default.yaml','r') as f:
            self.defaults = yaml.safe_load(f) #load(file,Loader=Loader)
        try:
            with open(file,'r') as f:
                self.settings = yaml.safe_load(f) #load(file,Loader=Loader)
                # print(self.settings)
        except FileNotFoundError:
            self.settings = self.defaults
            print("Run settings not found, reverting to defaults")
            
        self.set_gp_settings()
        self.set_acq_settings()
        self.set_ns_settings()
        self.set_optimizer_settings()
        self.set_bo_settings()

    def set_gp_settings(self):
        #method check
        method = self.set_method("GP")
        self.gp_settings = {}
        self.gp_settings[method] = self.set_from_file("GP",method)

    def set_ns_settings(self):
        #method 
        method = self.set_method("NS")
        self.ns_settings = {}
        self.ns_settings[method] = self.set_from_file("NS",method)

    def set_acq_settings(self):
        #method check
        method = self.set_method("ACQ")
        self.acq_settings = {}
        self.acq_settings[method] = self.set_from_file("ACQ",method)
    
    def set_optimizer_settings(self):
        method = self.set_method("optimizer")
        self.optimzer_settings = {}
        self.optimzer_settings[method] = self.set_from_file("optimizer",method)

    def set_bo_settings(self):
        method = self.set_method("BO")
        self.bo_settings = {}
        self.bo_settings[method] = self.set_from_file("BO",method)
    
    def set_method(self,module):
        if module in self.settings.keys():
            if 'method' in self.settings[module]['method']:
                method = self.settings[module]['method']
            else:
                method = self.defaults[module]['method']
        else:
            method = self.defaults[module]['method']
        return method

    def set_from_file(self,module,method:str):
        settings = {}
        if module in self.settings.keys():
            for key in self.defaults[module][method]:
                if key in self.settings[module][method]:
                    val = self.settings[module][method][key]
                else:
                    val = self.defaults[module][method][key]
                settings[key] = val
        else:
            settings = self.defaults[module][method]
        return settings