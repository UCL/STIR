# Demo how to use STIR from Python to access events in a list-mode file

# Copyright 2023 - University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

#%% Initial imports
import stir
import os

#%% read file
filename = '20170809_NEMA_60min_UCL.l.hdr'
# example using the hroot file in recon_test_pack, which needs some env variables to be set
#os.environ["INPUT_ROOT_FILE"]="test_PET_GATE.root"
#os.environ["EXCLUDE_SCATTERED"] = "1"
#os.environ["EXCLUDE_RANDOM"] = "0"
#filename = "root_header.hroot"

lm=stir.ListModeData.read_from_file(filename)
# could print some exam info
# print(lm.get_exam_info().parameter_info())
proj_data_info = lm.get_proj_data_info()
# could print some geometric info
#print(proj_data_info.parameter_info())

#%% loop over first few events and print some information
# create some variables for re-use in the loop
record = lm.get_empty_record()
bin = stir.Bin()
for i in range(50):
    lm.get_next_record(record)
    if (record.is_time()):
        print(f"Time: {record.time().get_time_in_millisecs()}")
    if (record.is_event()):
        event = record.event()
        lor = event.get_LOR()
        event.get_bin(bin, proj_data_info);
        # TODO We can will be able to simply print bin once STIR_TOF is on
        print(f"Event: p/d: {event.is_prompt()} LOR: {[lor.p1(), lor.p2()]}, ",
              f"bin: s:{bin.segment_num}, a: {bin.axial_pos_num}, v: {bin.view_num}, t:{bin.tangential_pos_num}")
