# Test file for STIR IO py.test
# Use as follows:
# on command line
#     py.test test_IO.py


#    Copyright (C) 2013 University College London
#    This file is part of STIR.
#
#    This file is free software; you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation; either version 2.1 of the License, or
#    (at your option) any later version.

#    This file is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    See STIR/LICENSE.txt for details

from stir import *
from stirextra import *
import os
# for Python2 and itertools.zip->zip (as in Python 3) 
try:
    import itertools.izip as zip
except ImportError:
    pass

def test_FloatVoxelsOnCartesianGrid(tmpdir):
    tmpdir.chdir()
    print("Creating files in ", os.getcwd())
    origin=FloatCartesianCoordinate3D(0,1,6)
    gridspacing=FloatCartesianCoordinate3D(1,1,2)
    minind=Int3BasicCoordinate(3)
    maxind=Int3BasicCoordinate(9)
    indrange=IndexRange3D(minind,maxind)
    image=FloatVoxelsOnCartesianGrid(indrange, origin,gridspacing)
    image[(5,3,4)]=2
    print(image.shape())
    output_format=InterfileOutputFileFormat()
    output_format.write_to_file("stir_python_test.hv", image) 
    image2=FloatVoxelsOnCartesianGrid.read_from_file('stir_python_test.hv')
    assert image.get_voxel_size()==image2.get_voxel_size()
    assert image.shape()==image2.shape()
    assert get_physical_coordinates_for_bounding_box(image) == get_physical_coordinates_for_bounding_box(image2)
    for i1,i2 in zip(image.flat(), image2.flat()):
        assert abs(i1-i2)<.01

def test_ProjDataInfo(tmpdir):
    tmpdir.chdir()
    print("Creating files in ", os.getcwd())
    s=Scanner.get_scanner_from_name("ECAT 962")
    #ProjDataInfoCTI(const shared_ptr<Scanner>& scanner_ptr,
    #		  const int span, const int max_delta,
    #             const int num_views, const int num_tangential_poss, 
    #
    examinfo=ExamInfo();
    projdatainfo=ProjDataInfo.ProjDataInfoCTI(s,3,6,8,6)
    assert projdatainfo.get_scanner().get_num_rings()==32
    projdata=ProjDataInterfile(examinfo, projdatainfo, "stir_python_test.hs")
    print(projdata.get_min_segment_num())
    print(projdata.get_max_segment_num())
    for seg in range(projdata.get_min_segment_num(), projdata.get_max_segment_num()+1):
        segment=projdatainfo.get_empty_segment_by_view(seg)
        segment.fill(seg+100)
        assert projdata.set_segment(segment)==Succeeded(Succeeded.yes)
    # now delete object such that the file gets closed and we can read it
    del projdata

    projdata2=ProjData.read_from_file('stir_python_test.hs');
    assert projdatainfo==projdata2.get_proj_data_info()
    for seg in range(projdata2.get_min_segment_num(), projdata2.get_max_segment_num()+1):
        # construct same segment data as above (TODO: better to stick it into a list or so)
        segment=projdatainfo.get_empty_segment_by_view(seg)
        segment.fill(seg+100)
        # read from file
        segment2=projdata2.get_segment_by_view(seg)
        # compare
        for i1,i2 in zip(segment.flat(), segment2.flat()):
            assert abs(i1-i2)<.01
    
