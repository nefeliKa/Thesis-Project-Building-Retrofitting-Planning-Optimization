from eppy import modeleditor
from eppy.modeleditor import IDF
import os
from geomeppy import IDF


def create_simulation(r_new_conductivity, w_new_conductivity,f_new_conductivity):


    iddfile = 'C:\EnergyPlusV23-2-0\Energy+.idd'
    fname1 = r"C:\Users\Nefeli\Desktop\Sim\unnamed\openstudio\run\in.idf"


    IDF.setiddname(iddfile)
    idf = IDF(fname1)

    # Import epw data
    idf.epw = (r"C:\Users\Nefeli\Downloads\NLD_NB_Eindhoven.AP.063700_TMYx(1)\NLD_NB_Eindhoven.AP.063700_TMYx.epw")

    # heating = idf.newidfobjects['HVACTEMPLATE:THERMOSTAT']
    # print(heating)
    idf.idfobjects['OUTPUTCONTROL:SIZING:STYLE']
    outputs = idf.idfobjects['OUTPUT:VARIABLE']
    # print(outputs)
    idf.idfobjects['BUILDINGSURFACE:DETAILED']
    idf.idfobjects['WINDOWMATERIAL:GAS']
    idf.idfobjects['AIRFLOWNETWORK:MULTIZONE:SURFACE:EFFECTIVELEAKAGEAREA']
    idf.idfobjects['AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK']
    idf.idfobjects['RUNPERIOD']
    idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
    idf.idfobjects['BUILDINGSURFACE:DETAILED']
    idf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['SCHEDULE:YEAR']
    idf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['SCHEDULE:YEAR']

    m = idf.idfobjects['MATERIAL']
    print(m)


    Roof_insulation = idf.idfobjects['MATERIAL'][17]
    Floor_insulation = idf.idfobjects['MATERIAL'][14]
    Facade_insulation = idf.idfobjects['MATERIAL'][1]

    r_original_conductivity =  idf.idfobjects['MATERIAL'][17]['Conductivity']
    f_original_conductivity =  idf.idfobjects['MATERIAL'][14]['Conductivity']
    w_original_conductivity =  idf.idfobjects['MATERIAL'][1]['Conductivity']
    roof_conductivity = (r_original_conductivity*r_new_conductivity)+r_original_conductivity
    wall_conductivity = (w_original_conductivity*w_new_conductivity)+w_original_conductivity
    floor_conductivity = (f_original_conductivity*f_new_conductivity)+ f_original_conductivity

    Roof_insulation.Conductivity = roof_conductivity
    Floor_insulation.Conductivity = floor_conductivity
    Facade_insulation.Conductivity = wall_conductivity


    output_dir = r'C:\Users\Nefeli\Desktop\Sim\unnamed\openstudio\run\New_folder'
    simulations_dir = os.path.join(output_dir, "simulations")
    file_name = f"roof({r_new_conductivity})_wall({w_new_conductivity})_floor({f_new_conductivity})"
    trial_simulations_dir = os.path.join(output_dir, "trial_simulation")

    # Run simulation
    idf.run(output_directory=trial_simulations_dir, output_prefix=file_name, expandobjects=True, annual=True)

    return trial_simulations_dir, file_name 



degradation_list = [5, 20, 35]
insulation_r = [i / 100 for i in degradation_list]
insulation_w = [i / 100 for i in degradation_list]
insulation_f = [i / 100 for i in degradation_list]
insulation_window = [i / 100 for i in degradation_list]

file_name_list = []
for r_deg_percentage in insulation_r: 
    for w_deg_percentage in insulation_w:
        for f_deg_percentage in insulation_f:
            filepath, result = create_simulation(r_deg_percentage, w_deg_percentage, f_deg_percentage)
            file_name_list.append(result)


# insulation_r = 0.1
# insulation_w = 0.1
# insulation_f = 0.1
# insulation_window = 0.1

# filepath, result = create_simulation(0.1, 0.1, 0.1, 0.1)

# # Convert list to string
list_as_string = ', '.join(map(str, file_name_list))
list_as_string = ', '.join(map(str, 'tryout'))
# Specify the file path
file_path = 'my_list.txt'

# Write the string to the file
with open(file_path, 'w') as file:
    file.write(list_as_string)   
