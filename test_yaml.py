from init import input_settings
settings = input_settings('../input_example.yaml')

# for key,val in settings.gp_settings.items():
#     print(key,"=",val)

# for key,val in settings.ns_settings.items():
#     print(key,"=",val)

# for key,val in settings.acq_settings.items():
#     print(key,"=",val)

# for key,val in settings.optimzer_settings.items():
#     print(key,"=",val)

print("GP",settings.gp_settings)

print("NS",settings.ns_settings)

print("ACQ",settings.acq_settings)

print("Optim",settings.optimzer_settings)

print("BO",settings.bo_settings)