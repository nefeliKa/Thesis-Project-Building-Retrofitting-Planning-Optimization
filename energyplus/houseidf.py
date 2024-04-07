from geomeppy import IDF
from eppy import modeleditor
import sys
import pandas as pd
import esoreader
import eppy
import os

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



# # Geometry
idf.add_block(
    name='House',
    coordinates=[(15, 0), (15, 10), (0, 15), (0, 0)],
    height=9,
    num_stories=3,
    below_ground_stories=1,
    below_ground_storey_height=2.5,
    zoning='by_storey')
idf.intersect_match()

# Show 3D model
# idf.view_model()
#windows 
idf.set_wwr(wwr=0.2)

idf.set_default_constructions()
# print(idf.idfobjects['CONSTRUCTION'])

##############Change info about the house

# # Change glazing material
# material_info = idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"][0]
# material_info.Name = "Glass_Material"
# material_info.UFactor = 2.5
# material_info.Solar_Heat_Gain_Coefficient = 0.8
# material_info.Visible_Transmittance = 0.8





# # Set Glazing Contruction 
# window_contruction = idf.idfobjects['CONSTRUCTION']
# window_contruction.Outside_Layer = "Glass_Material"
# print(window_contruction)

# #set ceiling construcction and material
# materials_list = idf.idfobjects['MATERIAL']
# get_original_material = idf.idfobjects['MATERIAL'][0]
# new_material = idf.copyidfobject(get_original_material)
# new_material.Name = "ExteriorInsulation"
# new_material.Conductivity = 5
# new_material.Thickness = 0.1
# # materials_list.append(new_material)

# ceiling_construction = idf.idfobjects['CONSTRUCTION'][-3]
# window_contruction.Outside_Layer = "DefaultMaterial"
# idf.newidfobject("CONSTRUCTION", Name = "Roof_Structure",Outside_Layer = 'DefaultMaterial', Layer_2 = "ExteriorInsulation")
# # print(ceiling_construction)

# Change surfaces
#######Windows
window_materials =  idf.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM']
# print(window_materials)
brick_material = idf.newidfobject('WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM', Name = "Glazings", UFactor='2', Solar_Heat_Gain_Coefficient = 0.7,Visible_Transmittance = 0.8 )
idf.newidfobject("CONSTRUCTION", Name = "Glazing_System",Outside_Layer = 'Glazings')
del window_materials[0]
# print('new')
# print(window_materials)


#Insutlation 
insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorInsulation", Thickness = 0.2,Specific_Heat = 1000, Conductivity = 0.5, Density = 600, Roughness = 'Smooth')
#######Roofs
brick_material = idf.newidfobject('MATERIAL', Name = "Wooden_Roof", Thickness = 0.2,Specific_Heat = 1000, Conductivity = 0.5, Density = 600, Roughness = 'Rough'  )
idf.newidfobject("CONSTRUCTION", Name = "Roof_Structure",Outside_Layer = 'Wooden_Roof', Layer_2 = "ExteriorInsulation")

#######Walls 
brick_material = idf.newidfobject('MATERIAL', Name = "Brick_Wall", Thickness = 0.2,Specific_Heat = 1000, Conductivity = 0.5, Density = 600, Roughness = 'Rough'  )
idf.newidfobject("CONSTRUCTION", Name = "Wall_Structure",Outside_Layer = 'Brick_wall', Layer_2 = "ExteriorInsulation")
# Change surfaces
#Surface types
surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
s_types = [surface.Surface_Type for surface in surfaces]
print(s_types)

materials =  idf.idfobjects["MATERIAL"]
print(materials)


#######Floors
brick_material = idf.newidfobject('MATERIAL', Name = "Concrete_Floor", Thickness = 0.2,Specific_Heat = 1000, Conductivity = 0.5, Density = 600, Roughness = 'Rough'  )
idf.newidfobject("CONSTRUCTION", Name = "Floor_Structure",Outside_Layer = 'Concrete_Floor', Layer_2 = "ExteriorInsulation")
# Change surfaces
#Surface types
surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
s_types = [surface.Surface_Type for surface in surfaces]
print(s_types)

#Surface types
surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
s_types = [surface.Surface_Type for surface in surfaces]
# print(s_types)

#Fenestration surfaces
fenestration = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
del idf.idfobjects['CONSTRUCTION'][6]
# print(fenestration)

#Surface Boundary Conditions
s_bcs = [surface.Outside_Boundary_Condition for surface in surfaces]
ext_walls = [surface for surface in surfaces if surface.Surface_Type =='wall']
floors = [surface for surface in surfaces if surface.Surface_Type =='floor']
roofs = [surface for surface in surfaces if surface.Surface_Type =='roof']
#set to adiabatic if interior wall or wall adjacent to another house
windows = [window for window in fenestration]


#Assign Constructions to Surfaces
for wall in ext_walls:
    wall.Construction_Name = "Wall_Structure"
for floor in floors:
    floor.Construction_Name = "Floor_Structure" # have to create separate floor constructions if multiple storeys
for roof in roofs:
    roof.Construction_Name = "Roof_Structure"
for window in windows: 
    window.Construction_Name = "Glazing_System"


# fenestration = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
print(fenestration[0])
print(fenestration)

# Input Site information
site_info = idf.idfobjects['SITE:LOCATION'][0]
print(site_info)
site_info.Name = 'Amsterdam.Port'
site_info.Latitude = 57
site_info.Longitude = 6
site_info.Time_Zone = -1
site_info.Elevation = 100

# Simulation Control
SimC = idf.idfobjects['SIMULATIONCONTROL'][0]
SimC.Do_Zone_Sizing_Calculation = 'Yes'
SimC.Do_System_Sizing_Calculation = 'No'
SimC.Do_Plant_Sizing_Calculation = 'No'
SimC.Run_Simulation_for_Weather_File_Run_Periods = 'Yes'

# Assign
stat = idf.newidfobject("HVACTEMPLATE:THERMOSTAT", Name="Zone Stat", Constant_Heating_Setpoint=20,
                        Constant_Cooling_Setpoint=25)
for zone in idf.idfobjects["ZONE"]:
    idf.newidfobject("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM", Zone_Name=zone.Name, Template_Thermostat_Name=stat.Name)

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

# Clean objects
idf.idfobjects.clear()

########################################SIMULATION######################################

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

headers = ["Heat", "Cool", "Total"]
header_format = "{:>10}" * (len(headers))
row_format = "{:>10.1f}" * (len(headers))
print(header_format.format(*headers))
for row in results:
    print(row_format.format(*row))


