#from eppy.modeleditor import IDF
from geomeppy import IDF
from eppy import modeleditor 
import sys
import pandas as pd
import esoreader
import eppy
import os

########################################CONFIGURATION######################################
#Set the idd file and idf file path
iddfile = 'C:\EnergyPlusV23-2-0\Energy+.idd'
fname1 = 'C:\EnergyPlusV23-2-0\Toy_files\Minimal_1 - Copy (2).idf'
IDF.setiddname(iddfile)

#idf1 now holds all the data to your in you idf file.
idf = IDF(fname1)

#Import epw data
idf.epw = r"C:\Users\Nefeli\Desktop\thesis_work\Codes\Thesis_Project\Weather_files\NLD_Amsterdam.062400_IWEC\NLD_Amsterdam.062400_IWEC.epw"

# # Geometry
idf.add_block(
    name = 'House',
    coordinates = [(15,0),(15,10),(0,15),(0,0)],
    height=9,
    num_stories = 3,
    below_ground_stories = 1,
    below_ground_storey_height = 2.5,
    zoning = 'by_storey')
idf.intersect_match()

# #Make 3d geometry
idf.set_default_constructions()

# Show 3D model
idf.view_model()

idf.Name = 'Small House in Amsterdam'
idf.Terrain = 'City'
idf.North_Axis = 30.0

#Change simple glazing
material_info = idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"][0]
material_info.UFactor = 2.5
material_info.Solar_Heat_Gain_Coefficient = 0.8
material_info.Visible_Transmittance = 0.8
print(material_info)

#Input Site information
site_info = idf.idfobjects['SITE:LOCATION'][0]
print(site_info)
site_info.Name = 'Amsterdam.Port'
site_info.Latitude = 57
site_info.Longitude = 6
site_info.Time_Zone = -1
site_info.Elevation = 100

#Simulation Control
SimC = idf.idfobjects['SIMULATIONCONTROL'][0]
SimC.Do_Zone_Sizing_Calculation= 'Yes'
SimC.Do_System_Sizing_Calculation= 'No'
SimC.Do_Plant_Sizing_Calculation= 'No'
SimC.Run_Simulation_for_Weather_File_Run_Periods= 'Yes'


#Assign 
stat = idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name="Zone Stat", Constant_Heating_Setpoint=20, Constant_Cooling_Setpoint=25)
for zone in idf.idfobjects["ZONE"]: 
     idf.newidfobject("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", Zone_Name=zone.Name, Template_Thermostat_Name=stat.Name)

idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Utility Use Per Conditioned Floor Area")
idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Heating Energy", Reporting_Frequency="Hourly")
idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Cooling Energy", Reporting_Frequency="Hourly")

output_dir = r'C:\Users\Nefeli\Desktop\thesis_work\Codes\Thesis_Project\TrialFiles\Output'
simulations_dir = os.path.join(output_dir, "simulations")
# Run save configuration
idf.save(filename=os.path.join(output_dir, "config.idf"), lineendings='default', encoding='latin-1')
idf.run(output_directory=simulations_dir, expandobjects=True,annual=True)


########################################SIMULATION######################################
#Read results
from eppy.results import readhtml
import pprint

#  Helper class to extract total energy use in kWh, via the esopackage
class ESO:
     def __init__(self, path):
          self.dd, self.data = esoreader.read(path)
     
     def read_var(self, variable, frequency="Hourly"):
            return [
                {"key": k, "series": self.data[self.dd.index[frequency, k, variable]]}
                for _f, k, _v in self.dd.find_variable(variable)
            ]
     
     def total_kwh(self, variable, frequency="Hourly"):
            j_per_kwh = 3_600_000
            results = self.read_var(variable, frequency)
            return sum(sum(s["series"]) for s in results) / j_per_kwh
     

########################################POST PROCESSING######################################
results = [] 
eso = ESO(os.path.join(simulations_dir, "eplusout.eso"))
heat = eso.total_kwh("Zone Ideal Loads Supply Air Total Heating Energy")
cool = eso.total_kwh("Zone Ideal Loads Supply Air Total Cooling Energy")
results.append([heat, cool, heat + cool])
# idf.run( )

headers = [ "Heat", "Cool", "Total"]
header_format = "{:>10}" * (len(headers))
row_format = "{:>10.1f}" * (len(headers))
print(header_format.format(*headers))
for row in results:
     print(row_format.format(*row))

#Clean objects
idf.idfobjects.clear()