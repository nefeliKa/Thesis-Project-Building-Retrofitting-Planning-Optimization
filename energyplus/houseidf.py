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

#create all possible scenarios
#roof insulation, wall insulation, floor insulation 
#apply these scenarios to the insulation 
#save the results 



########################################CONFIGURATION######################################
def create_simulation(r_new_conductivity, w_new_conductivity,f_new_conductivity):
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

    #windows position and wwr
    idf.set_wwr(wwr=0.35, orientation= 'south', wwr_map={0:0} )
    idf.set_wwr(wwr=0.35,orientation= 'north', wwr_map={0:0} )

    # Show 3D model
    # idf.view_model()

    ##############Change info about the house
    #######Windows
    window_materials =  idf.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM']
    # print(window_materials)
    brick_material = idf.newidfobject('WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM', Name = "Glazings", UFactor='2.8', Solar_Heat_Gain_Coefficient = 0.3,Visible_Transmittance = 0.8 )
    idf.newidfobject("CONSTRUCTION", Name = "Glazing_System",Outside_Layer = 'Glazings')
    # del window_materials[0]
    # print(window_materials)

    #######Add Insulation Material 
    original_conductivity =  0.35
    roof_conductivity = (original_conductivity*r_new_conductivity)+original_conductivity
    wall_conductivity = (original_conductivity*w_new_conductivity)+original_conductivity
    floor_conductivity = (original_conductivity*f_new_conductivity)+ original_conductivity
    roof_insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorRoofInsulation", Thickness = 0.05,Specific_Heat = 1400, Conductivity = roof_conductivity, Density = 25, Roughness = 'Smooth')
    wall_insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorWallInsulation", Thickness = 0.05,Specific_Heat = 1400, Conductivity = wall_conductivity, Density = 25, Roughness = 'Smooth')
    groundfloor_insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorFloorInsulation", Thickness = 0.05,Specific_Heat = 1400, Conductivity = floor_conductivity, Density = 25, Roughness = 'Smooth')
    # roof_insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorRoofInsulation", Thickness = 0.05,Specific_Heat = 1400, Conductivity = original_conductivity, Density = 25, Roughness = 'Smooth')
    # wall_insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorWallInsulation", Thickness = 0.05,Specific_Heat = 1400, Conductivity = original_conductivity, Density = 25, Roughness = 'Smooth')
    # groundfloor_insulation_material = idf.newidfobject('MATERIAL',  Name = "ExteriorFloorInsulation", Thickness = 0.05,Specific_Heat = 1400, Conductivity = original_conductivity, Density = 25, Roughness = 'Smooth')



    #######Roofs
    #Plywood roof
    roof_material = idf.newidfobject('MATERIAL', Name = "Wooden_Roof", Thickness = 0.1,Specific_Heat = 1460, Conductivity = 0.15, Density = 700, Roughness = 'Rough'  )
    idf.newidfobject("CONSTRUCTION", Name = "Roof_Structure",Outside_Layer = 'Wooden_Roof', Layer_2 = "ExteriorRoofInsulation")

    #######Walls 
    brick_material = idf.newidfobject('MATERIAL', Name = "Brick_Wall", Thickness = 0.1,Specific_Heat = 800, Conductivity = 0.84, Density = 1700, Roughness = 'Rough'  )
    idf.newidfobject("CONSTRUCTION", Name = "Wall_Structure",Outside_Layer = 'Brick_wall', Layer_2 = "ExteriorWallInsulation",Layer_3 = 'Brick_wall')
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
    idf.newidfobject("CONSTRUCTION", Name = "Floor_Structure",Outside_Layer = 'Concrete_Floor', Layer_2 = "ExteriorFloorInsulation")

    # Change surfaces
    #Surface types
    surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
    s_types = [surface.Surface_Type for surface in surfaces]

    #Fenestration surfaces
    fenestration = idf.idfobjects['FENESTRATIONSURFACE:DETAILED']

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
    site_info.Latitude = 52.372049
    site_info.Longitude = 4.880162
    site_info.Time_Zone = +2
    site_info.Elevation = 0

    # Simulation Control
    simulation = idf.idfobjects['SIMULATIONCONTROL']
    SimC = idf.idfobjects['SIMULATIONCONTROL'][0]
    SimC.Do_Zone_Sizing_Calculation = 'Yes'
    SimC.Do_System_Sizing_Calculation = 'No'
    SimC.Do_Plant_Sizing_Calculation = 'No'
    SimC.Run_Simulation_for_Weather_File_Run_Periods = 'Yes'

    stat = idf.newidfobject("HVACTEMPLATE:THERMOSTAT",Name="Zone Stat", Constant_Heating_Setpoint=20,Constant_Cooling_Setpoint=25,)

    # heatpump = idf.newidfobject("HVACTEMPLATE:ZONE:WATERTOAIRHEATPUMP")
    # heatpump.Heat_Pump_Heating_Coil_Gross_Rated_COP =3 
    # heatpump.Template_Thermostat_Name = 'Zone Stat'

    for zone in idf.idfobjects["ZONE"]:
            idf.newidfobject("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM",Zone_Name=zone.Name,Template_Thermostat_Name=stat.Name,)


    idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Utility Use Per Conditioned Floor Area")
    idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Heating Energy",
                    Reporting_Frequency="Hourly")
    idf.newidfobject("OUTPUT:VARIABLE", Variable_Name="Zone Ideal Loads Supply Air Total Cooling Energy",
                    Reporting_Frequency="Hourly")

    import os
    from eppy.results import readhtml

    output_dir = r'C:\Users\Nefeli\Desktop\thesis_work\Codes\Thesis_Project\TrialFiles\Output'
    simulations_dir = os.path.join(output_dir, "simulations")
    file_name = f"roof({r_new_conductivity})_wall({w_new_conductivity})_floor({f_new_conductivity})"
    trial_simulations_dir = os.path.join(output_dir, "trial_simulation")

    # Run simulation
    idf.run(output_directory=trial_simulations_dir, output_prefix=file_name, expandobjects=True, annual=True)

    return trial_simulations_dir, file_name 

degradation_list = [10, 30, 50]
insulation_r = [i / 100 for i in degradation_list]
insulation_w = [i / 100 for i in degradation_list]
insulation_f = [i / 100 for i in degradation_list]

file_name_list = []
for r_deg_percentage in insulation_r: 
    for w_deg_percentage in insulation_w:
        for f_deg_percentage in insulation_f:
            filepath, result = create_simulation(r_deg_percentage, w_deg_percentage, f_deg_percentage)
            file_name_list.append(result)
            
# Convert list to string
list_as_string = ', '.join(map(str, file_name_list))

# Specify the file path
file_path = 'my_list.txt'

# Write the string to the file
with open(file_path, 'w') as file:
    file.write(list_as_string)                   


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



#     # Path to the .tbl results file
#     tbl_file_path = os.path.join(simulations_dir, f"{file_name}.tbl")

#     # Check if the .tbl file exists
#     if os.path.exists(tbl_file_path):
#         # Read .tbl file
#         with open(tbl_file_path, 'r') as file:
#             fhandle = file.read()
#             htables = readhtml.titletable(fhandle)
#             site_energy_t = htables[0][1]
#             net_energy_mj = site_energy_t[2][3]
#             net_energy_kwh = net_energy_mj * 0.28
#             # return net_energy_kwh

# class ESO:
#         def __init__(self, path):
#             self.dd, self.data = esoreader.read(path)
        
#         def read_var(self, variable, frequency="Hourly"):
#             return [
#                 {"key": k, "series": self.data[self.dd.index[frequency, k, variable]]}
#                 for _f, k, _v in self.dd.find_variable(variable)
#             ]
        
#         def total_kwh(self, variable, frequency="Hourly"):
#             j_per_kwh = 3_600_000
#             results = self.read_var(variable, frequency)
#             return sum(sum(s["series"]) for s in results) / j_per_kwh
        

# results = []
# for in 
#     eso = ESO(f"tests/tutorial/{north}_{south}_out.eso")
#     heat = eso.total_kwh("Zone Ideal Loads Supply Air Total Heating Energy")
#     cool = eso.total_kwh("Zone Ideal Loads Supply Air Total Cooling Energy")




# trial_simulations_dir = os.path.join(output_dir, "trial_simulation")
# degradation_list = [10,20,40]
# insultation_r = [i / 100 for i in degradation_list]
# print(insultation_r)
# insultation_w = [i / 100 for i in degradation_list]
# print(insultation_w)
# insultation_f = [i / 100 for i in degradation_list]
# print(insultation_f)

# for r_deg_percentage in insultation_r(): 
#         for w_deg_percentage in insultation_w():
#              for f_deg_percentage in insultation_f():
#                 create_simulation()

# from eppy.results import readhtml
# import pprint

# fresults = os.path.join(simulations_dir,"eplustbl.htm")
# fhandle = open(fresults, 'r').read()
# htables = readhtml.titletable(fhandle)
# site_energy_t = htables[0][1]
# pp = pprint.PrettyPrinter()
# net_energy_mj = site_energy_t[2][3]
# net_energy_kwh = net_energy_mj*0.28
# print(net_energy_kwh)




