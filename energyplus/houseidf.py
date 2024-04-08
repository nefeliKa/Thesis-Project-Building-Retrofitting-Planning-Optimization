from geomeppy import IDF
from eppy import modeleditor
import sys
import pandas as pd
import esoreader
import eppy
import os



#TODO: MAKE LOOPS AND DATAFRAME, SIMULATE ALL SCENARIOS, ADD ACTUAL MATERIAL DATA, CHECK THE WINDOWS, CHECK THE HVAC
#material properties from "https://help.iesve.com/ve2021/table_6_thermal_conductivity__specific_heat_capacity_and_density.htm"


#Iterate through all possible combinations
# insulation conductivity = [0.1,0.2,0.3]



########################################CONFIGURATION######################################
# Set the idd file and idf file path
iddfile = 'C:\EnergyPlusV23-2-0\Energy+.idd'
fname1 = 'C:\EnergyPlusV23-2-0\Toy_files\Minimal_1 - Copy (2).idf'
IDF.setiddname(iddfile)

# idf1 now holds all the data to your in you idf file.
idf = IDF(fname1)

# Import epw data
idf.epw = (r"C:\Users\Nefeli\Desktop\thesis_work\Codes\Thesis_Project\Weather_files\NLD_Amsterdam.062400_IWEC"
           r"\NLD_Amsterdam.062400_IWEC.epw")

# Clean objects
# idf.idfobjects.clear()
block = idf.block
del block

# # Geometry
idf.add_block(
    name='House',
    coordinates=[(0, 0), (5.40,0), (5.40, 8.25), (0, 8.25)],
    height=9,
    num_stories=3,
    below_ground_stories=1,
    below_ground_storey_height=2.5,
    zoning='by_storey')
idf.intersect_match()

# idf.add_zone(
#     name = "All floors", 

# )
#windows position and wwr
idf.set_wwr(wwr=0.35, orientation= 'south', wwr_map={0:0} )
idf.set_wwr(wwr=0.35,orientation= 'north', wwr_map={0:0} )


# Show 3D model
idf.view_model()

# print(idf.idfobjects['CONSTRUCTION'])

##############Change info about the house
# Change surfaces
#######Windows
window_materials =  idf.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM']
# print(window_materials)
brick_material = idf.newidfobject('WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM', Name = "Glazings", UFactor='2.8', Solar_Heat_Gain_Coefficient = 0.3,Visible_Transmittance = 0.8 )
idf.newidfobject("CONSTRUCTION", Name = "Glazing_System",Outside_Layer = 'Glazings')
# del window_materials[0]
print(window_materials)


#######Add Insulation Material 
insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorInsulation", Thickness = 0.05,Specific_Heat = 1400, Conductivity = 0.035, Density = 25, Roughness = 'Smooth')

#######Roofs
#Plywood roof
roof_material = idf.newidfobject('MATERIAL', Name = "Wooden_Roof", Thickness = 0.1,Specific_Heat = 1460, Conductivity = 0.15, Density = 700, Roughness = 'Rough'  )
idf.newidfobject("CONSTRUCTION", Name = "Roof_Structure",Outside_Layer = 'Wooden_Roof', Layer_2 = "ExteriorInsulation")

#######Walls 
brick_material = idf.newidfobject('MATERIAL', Name = "Brick_Wall", Thickness = 0.1,Specific_Heat = 800, Conductivity = 0.84, Density = 1700, Roughness = 'Rough'  )
idf.newidfobject("CONSTRUCTION", Name = "Wall_Structure",Outside_Layer = 'Brick_wall', Layer_2 = "ExteriorInsulation",Layer_3 = 'Brick_wall')
# Change surfaces
#Surface types
surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
s_types = [surface.Surface_Type for surface in surfaces]
# print(s_types)

materials =  idf.idfobjects["MATERIAL"]
# print(materials)

#Ceilings
false_ceiling = idf.newidfobject('MATERIAL', Name = "False_Ceiling", Thickness = 0.2,Specific_Heat = 1000, Conductivity = 0.5, Density = 600, Roughness = 'Rough'  )
idf.newidfobject("CONSTRUCTION", Name = "Ceiling_Structure",Outside_Layer = 'False_Ceiling')

#######Floors
brick_material = idf.newidfobject('MATERIAL', Name = "Concrete_Floor", Thickness = 0.2,Specific_Heat = 1000, Conductivity = 0.5, Density = 600, Roughness = 'Rough'  )
idf.newidfobject("CONSTRUCTION", Name = "Floor_Structure",Outside_Layer = 'Concrete_Floor', Layer_2 = "ExteriorInsulation")

# Change surfaces
#Surface types
surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
s_types = [surface.Surface_Type for surface in surfaces]

#Fenestration surfaces
fenestration = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
# del idf.idfobjects['CONSTRUCTION'][6]
# print(fenestration)


#Surface Boundary Conditions
s_bcs = [surface.Outside_Boundary_Condition for surface in surfaces]
ext_walls = [surface for surface in surfaces if surface.Surface_Type =='wall']
floors = [surface for surface in surfaces if surface.Surface_Type =='floor']
ceilings = [surface for surface in surfaces if surface.Surface_Type =='ceiling']
roofs = [surface for surface in surfaces if surface.Surface_Type =='roof']
#set to adiabatic if interior wall or wall adjacent to another house
windows = [window for window in fenestration]


#Assign Constructions to Surfaces
for floor in floors:
    floor.Construction_Name = "Floor_Structure" # have to create separate floor constructions if multiple storeys
for roof in roofs:
    roof.Construction_Name = "Roof_Structure"
for window in windows: 
    window.Construction_Name = "Glazing_System"
for ceiling in ceilings: 
    ceiling.Construction_Name = "Ceiling_Structure"

# Filter out walls without windows and make them adiabatic
surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
fenestration = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']  # Assuming you have fenestration surfaces defined

for surface in surfaces:
    if surface.Surface_Type == "wall":
        windw = 'window'
        surface_name = surface.Name + ' ' + windw
        # print(surface_name)
        #HEre check if the fenestration and the surface have similar
        for fen in fenestration: 
            if fen.Building_Surface_Name == surface_name :
                 surface.Construction_Name = "Wall_Structure"
                 surface.Outside_Boundary_Condition = "Adiabatic"
            else :
                 surface.Construction_Name = "Wall_Structure"


# Input Site information
site_info = idf.idfobjects['SITE:LOCATION'][0]
# print(site_info)
site_info.Name = 'Amsterdam.Port'
site_info.Latitude = 57
site_info.Longitude = 6
site_info.Time_Zone = -1
site_info.Elevation = 100

# Simulation Control
simulation = idf.idfobjects['SIMULATIONCONTROL']
SimC = idf.idfobjects['SIMULATIONCONTROL'][0]
SimC.Do_Zone_Sizing_Calculation = 'Yes'
SimC.Do_System_Sizing_Calculation = 'No'
SimC.Do_Plant_Sizing_Calculation = 'No'
SimC.Run_Simulation_for_Weather_File_Run_Periods = 'Yes'

stat = idf.newidfobject("HVACTEMPLATE:THERMOSTAT",Name="Zone Stat", Constant_Heating_Setpoint=20,Constant_Cooling_Setpoint=25,)
for zone in idf.idfobjects["ZONE"]:
      idf.newidfobject("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM",Zone_Name=zone.Name,Template_Thermostat_Name=stat.Name,)

# print(therm)
# # print(idf.idfobjects['ZONE'])
# for zone in idf.idfobjects["ZONE"]:
#     zonename = zone.Name 
#     idf.newidfobject('ZONEINFILTRATION:EFFECTIVELEAKAGEAREA',  Name = 'Zone Infiltration', Schedule_Name = 'INF-SCHED', Effective_Air_Leakage_Area = 500, Stack_Coefficient = 0.000145, Wind_Coefficient = 0.000174)

# BL = idf.idfobjects['HVACTEMPLATE:PLANT:BOILER']

# print(BL)

# zoneinfiltration = idf.idfobjects['ZONEINFILTRATION:EFFECTIVELEAKAGEAREA']
# del zoneinfiltration

idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Utility Use Per Conditioned Floor Area")
idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Heating Energy",
                 Reporting_Frequency="Hourly")
idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Cooling Energy",
                 Reporting_Frequency="Hourly")


output_dir = r'C:\Users\Nefeli\Desktop\thesis_work\Codes\Thesis_Project\TrialFiles\Output'
simulations_dir = os.path.join(output_dir, "simulations")
# Run save configuration
idf.save(filename=os.path.join(output_dir, "config.idf"), lineendings='default', encoding='latin-1')
idf.run(output_directory=simulations_dir, expandobjects=True, annual=True)



########################################SIMULATION######################################

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


# #######################################POST PROCESSING######################################
# results = []
# eso = ESO(os.path.join(simulations_dir, "eplusout.eso"))
# heat = eso.total_kwh("Zone Ideal Loads Supply Air Total Heating Energy")
# cool = eso.total_kwh("Zone Ideal Loads Supply Air Total Cooling Energy")
# results.append([heat, cool, heat + cool])
# # idf.run( )

# headers = ["Heat", "Cool", "Total"]
# header_format = "{:>10}" * (len(headers))
# row_format = "{:>10.1f}" * (len(headers))
# print(header_format.format(*headers))
# for row in results:
#     print(row_format.format(*row))

from eppy.results import readhtml
import pprint

fresults = os.path.join(simulations_dir,"eplustbl.htm")
fhandle = open(fresults, 'r').read()
htables = readhtml.titletable(fhandle)
site_energy_t = htables[0][1]
pp = pprint.PrettyPrinter()
net_energy_mj = site_energy_t[2][3]
net_energy_kwh = net_energy_mj*0.28
print(net_energy_kwh)




