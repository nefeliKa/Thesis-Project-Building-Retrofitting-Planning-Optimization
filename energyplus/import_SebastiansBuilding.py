from eppy import modeleditor
from eppy.modeleditor import IDF
import os
from geomeppy import IDF
iddfile = 'C:\EnergyPlusV23-2-0\Energy+.idd'
fname1 = r"C:\Users\Nefeli\Downloads\Base_model simple_schedule.idf"


IDF.setiddname(iddfile)
idf = IDF(fname1)

# Import epw data
idf.epw = (r"C:\Users\Nefeli\Desktop\thesis_work\Codes\Thesis_Project\Weather_files\NLD_Amsterdam.062400_IWEC"
            r"\NLD_Amsterdam.062400_IWEC.epw")

# heating = idf.newidfobjects['HVACTEMPLATE:THERMOSTAT']
# print(heating)
idf.idfobjects['OUTPUTCONTROL:SIZING:STYLE']
outputs = idf.idfobjects['OUTPUT:VARIABLE']
print(outputs)
idf.idfobjects['RUNPERIOD']
idf.idfobjects['FENESTRATIONSURFACE:DETAILED']
idf.idfobjects['BUILDINGSURFACE:DETAILED']
idf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['SCHEDULE:YEAR']
idf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['OUTPUTCONTROL:SIZING:STYLE'].theidf.idfobjects['SCHEDULE:YEAR']
idf.idfobjects['MATERIAL']
idf.idfobjects['CONSTRUCTION']
idf.idfobjects['BUILDINGSURFACE:DETAILED']
idf.idfobjects['WINDOWMATERIAL:GAS']
idf.idfobjects['AIRFLOWNETWORK:MULTIZONE:SURFACE:EFFECTIVELEAKAGEAREA']
idf.idfobjects['AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK']



output_dir = r'C:\Users\Nefeli\Desktop\thesis_work\Codes\Thesis_Project\TrialFiles\Output'
simulations_dir = os.path.join(output_dir, "importedfilesimulations")
file_name = 'test'
trial_simulations_dir = os.path.join(output_dir)
idf.run(output_directory= simulations_dir, annual=True,expandobjects=True)
