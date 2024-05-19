# from houseidf import file_name_list,filepath
import esoreader
import os
from eppy.results import readhtml
import pprint
import pandas as pd
import time

start_time = time.time()  # Record the start time

# #######################################SIMULATION######################################

# #  Helper class to extract total energy use in kWh, via the esopackage
# class ESO:
#     def __init__(self, path):
#         self.dd, self.data = esoreader.read(path)

#     def read_var(self, variable, frequency="Hourly"):
#         return [
#             {"key": k, "series": self.data[self.dd.index[frequency, k, variable]]}
#             for _f, k, _v in self.dd.find_variable(variable)
#         ]

#     def total_kwh(self, variable, frequency="Hourly"):
#         j_per_kwh = 3_600_000
#         results = self.read_var(variable, frequency)
#         return sum(sum(s["series"]) for s in results) / j_per_kwh


# # #######################################POST PROCESSING######################################
# results = []

# for name in file_name_list: 
#     eso = ESO(os.path.join(filepath, name,".eso"))
#     heat = eso.total_kwh("Zone Ideal Loads Supply Air Total Heating Energy")
#     cool = eso.total_kwh("Zone Ideal Loads Supply Air Total Cooling Energy")
#     results.append([heat, cool, heat + cool])
#     # idf.run( )

#     headers = ["Heat", "Cool", "Total"]
#     header_format = "{:>10}" * (len(headers))
#     row_format = "{:>10.1f}" * (len(headers))
#     print(header_format.format(*headers))
#     for row in results:
#         print(row_format.format(*row))


# Directory path
# output_directory = r"C:\Users\Nefeli\Desktop\Sim\unnamed\openstudio\run\New_folder\importedfilesimulations"
output_directory = r"C:\Users\Nefeli\Desktop\Sim\unnamed\openstudio\run\New_folder\trial_simulation"
os.makedirs(output_directory, exist_ok=True)

dataframe = pd.DataFrame(columns=['state', 'energy[kWh/m2]', 'energy[kWh]'])

import re
# regex for the parentheses, e.g go from this: roof(0.1)_wall(0.1)_floor(0.1)tbl.htm to [0.1, 0.1, 0.1]
pattern = r'\((\d+\.\d+)\)'

degradations2states = {
    "0.0": 0,
    "0.2": 1,
    "0.43": 2
}

# Iterate over all files in the directory
for file in os.listdir(output_directory):
    # Check if the file has a ".tbl" extension
    if file.endswith('tbl.htm'):
        # Print a message for files with "tbl" extension
        print(f"~~~~~~~~~~~~~~~~~~ Processing {file}")
        matches = re.findall(pattern, file)
        print(matches)
        current_state = (degradations2states[matches[0]], degradations2states[matches[1]], degradations2states[matches[2]])

        fresults = os.path.join(output_directory, file)
        fhandle = open(fresults, 'r').read()
        htables = readhtml.titletable(fhandle)
        site_energy_t = htables[0][1]## Get from 'Total Energy [GJ]'
        # print(site_energy_t)
        pp = pprint.PrettyPrinter()
        net_energy_mj = site_energy_t[2][2] ## Get the 'Net Site Energy'
        # print(net_energy_mj)
        building_area = htables[2][1][1][1] #Get building area [m2]
        # print(building_area)
        net_energy_kwh_m2 = net_energy_mj*0.28 # Convert megajoules to kWh
        net_energy_kwh = net_energy_kwh_m2 * building_area
        print(f"{net_energy_kwh_m2:.2f} kWh/m2") # print kWh/m2

        current_building_setup = {'state': current_state, 'energy[kWh/m2]': net_energy_kwh_m2, 'energy[kWh]': net_energy_kwh}
        dataframe = pd.concat([dataframe, pd.DataFrame([current_building_setup])], ignore_index=True)



dataframe.to_csv('building_scenarios.csv', index=False)


end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the execution time

print("Total execution time: {:.2f} seconds".format(execution_time))