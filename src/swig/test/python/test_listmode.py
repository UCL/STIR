# Test file for STIR listmode reading
# Use as follows:
# on command line
#     py.test test_listmode.py

# Copyright 2023 - University College London
# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

# This file checks the first 2 events in the recon_test_pack/test_PET_GATE.root

# %% Initial imports
import stir
import os

# %% read file
# location of script
loc = os.path.dirname(__file__)
# location of ROOT file
loc = os.path.join(loc, "..", "..", "..", "..", "recon_test_pack")
filename = "root_header.hroot"

os.environ["INPUT_ROOT_FILE"] = "test_PET_GATE.root"
os.environ["EXCLUDE_SCATTERED"] = "1"
os.environ["EXCLUDE_RANDOM"] = "0"


def test_get_record():
    try:
        lm = stir.ListModeData.read_from_file(os.path.join(loc, filename))
        proj_data_info = lm.get_proj_data_info()
    except RuntimeError:
        print(f"Could not open {filename}")
        print("ROOT support not enabled?")
        return

    record = lm.get_empty_record()
    b = stir.Bin()
    # first event
    lm.get_next_record(record)
    assert record.is_time()
    assert record.time().get_time_in_millisecs() == 2
    assert record.is_event()
    event = record.event()
    lor = event.get_LOR()
    event.get_bin(b, proj_data_info)
    # test coordinates allowing for a swap (as this is non-TOF data)
    known = stir.Float3BasicCoordinate((2.03125, -149.102, -299.989))
    diff1 = stir.FloatCartesianCoordinate3D(lor.p1() - known)
    diff2 = stir.FloatCartesianCoordinate3D(lor.p2() - known)
    assert (abs(diff2.x()) < 0.1 and abs(diff2.y()) < 0.1 and abs(diff2.z()) < 0.1) or (
        abs(diff1.x()) < 0.1 and abs(diff1.y()) < 0.1 and abs(diff1.z()) < 0.1
    )
    assert (
        b.segment_num == 1
        and b.axial_pos_num == 1
        and b.view_num == 112
        and b.tangential_pos_num == -102
    )
    # second event
    lm.get_next_record(record)
    assert record.is_time()
    assert record.time().get_time_in_millisecs() == 3
    assert record.is_event()
    event = record.event()
    lor = event.get_LOR()
    event.get_bin(b, proj_data_info)
    # test coordinates allowing for a swap (as this is non-TOF data)
    known = stir.Float3BasicCoordinate((-2.03125, 325.646, -78.6102))
    diff1 = stir.FloatCartesianCoordinate3D(lor.p1() - known)
    diff2 = stir.FloatCartesianCoordinate3D(lor.p2() - known)
    assert (abs(diff2.x()) < 0.1 and abs(diff2.y()) < 0.1 and abs(diff2.z()) < 0.1) or (
        abs(diff1.x()) < 0.1 and abs(diff1.y()) < 0.1 and abs(diff1.z()) < 0.1
    )
    assert (
        b.segment_num == -1
        and b.axial_pos_num == 1
        and b.view_num == 12
        and b.tangential_pos_num == -14
    )
