# Demo of how to use STIR from Python to construct some projection data from scratch

# Copyright 2023 - University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

#%% Initial imports
import stir

#%% check list of predefined scanners
print(stir.Scanner.get_names_of_predefined_scanners())

#%% create a scanner
scanner=stir.Scanner.get_scanner_from_name("Siemens mMR")

#%% create a ProjDataInfo, describing the geometry of the data acquired for that scanner
span = 11
max_ring_diff = 60
num_views = scanner.get_num_detectors_per_ring()//2
num_tangential_poss = scanner.get_default_num_arccorrected_bins()
proj_data_info = stir.ProjDataInfo.construct_proj_data_info(scanner, span, max_ring_diff, num_views, num_tangential_poss, False);

#%% create radionuclide
modality = stir.ImagingModality(stir.ImagingModality.PT)
db = stir.RadionuclideDB()
r = db.get_radionuclide(modality, "^18^Fluorine")

#%% supported values are determined by your radionuclide JSON database
with open(stir.find_STIR_config_file("radionuclide_names.json")) as f:
    print(f.read())

#%% create ExamInfo object
exam_info = stir.ExamInfo(modality)
exam_info.set_radionuclide(r)
exam_info.patient_position=stir.PatientPosition(stir.PatientPosition.HFS)
print(exam_info.parameter_info())

#%% create projection data in memory, filled with 1
proj_data = stir.ProjDataInMemory(exam_info, proj_data_info)
proj_data.fill(1)

#%% write it to file
# STIR can currently only write Interfile projection data
proj_data.write_to_file('ones.hs')
