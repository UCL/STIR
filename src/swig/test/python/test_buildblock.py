# Test file for STIR buildblock py.test
# Use as follows:
# on command line
#     py.test test_buildblock.py


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

try:
    import pytest
except ImportError:
    # No pytest, try older py.test
    try:
        import py.test as pytest
    except ImportError:
        raise ImportError('Tests require pytest or py<1.4')

from stir import *
import stirextra

def test_Vector():
    dv=FloatVector(3)
    dv[2]=1
    #for a in dv:
    #print a
    #assert dv[0]==0 #probably not initialised
    #assert dv[1]==0 #probably not initialised
    assert dv[2]==1

    # assign same vector to another variable
    dvcopy=dv;
    # this will modify the original vector as well (as they are the same)
    # this behaviour is the same as for a Python list, but is different from C++
    dvcopy[2]=4;
    assert dv[2]==dvcopy[2]

    # instead, in Python we need to explicitly create a new object 
    dvcopy=FloatVector(dv)
    dvcopy[2]=dv[2]+2;
    assert dv[2]+2 == dvcopy[2]

def test_Coordinate():
    #a=make_FloatCoordinate(1,2,3)
    a=Float3BasicCoordinate((1,2,3))
    assert a.__getitem__(1)==1
    assert a[2]==2

    a=Float3Coordinate(1,2,3)
    assert a[2]==2
    # use tuple constructor
    a=Float3BasicCoordinate((1,2,3))
    assert a[3]==3

def test_VectorWithOffset():
    v=FloatVectorWithOffset(1,3)
    v[2]=3
    assert v[2]==3
    #assert v[1]==0 #probably not initialised
    with pytest.raises(IndexError):
        v[0] # check index-out-of-range

def test_Array1D():
    a=FloatArray1D(1,5)
    a[2]=3
    assert a[2]==3
    #help(FloatArray1D)
    # create a copy
    a2=FloatArray1D(a);
    # this should not modify the original
    a2[2]=4
    assert a[2]==3
    assert a2[2]==4

def test_Array1Diterator():
    a=FloatArray1D(1,3)
    a[2]=3
    a[1]=2
    sum=0
    for el in a: sum+=el
    assert sum==a.sum()

def test_Array2D():
    a2=FloatArray2D(IndexRange2D(Int2BasicCoordinate((3,3)), Int2BasicCoordinate((9,9))))
    a2.fill(2);
    ind=Int2BasicCoordinate((4,5))
    assert a2[ind]==2
    a2[ind]=4
    assert a2[ind]==4
    assert a2[(4,5)]==4
    #assert a2[ind[1]][ind[2]]==4
    # let's set a whole row
    #a1=a2[4]
    #a1[5]=66;
    #print 'original value in 2D array should be 2:', a2[make_IntCoordinate(4,5)], ', but the value in the copied row has to be 66:', a1[5]
    #a2[4]=a1
    #print 'now the entry in the 2D array has to be modified to 66 as well:', a2[Int2BasicCoordinate((4,5))]

def test_Array2Diterator():
    a2=FloatArray2D(IndexRange2D(Int2BasicCoordinate((3,3)), Int2BasicCoordinate((9,9))))
    a2.fill(2);
    # use flat iterator, i.e. go over all elements
    assert a2.sum() == sum(a2.flat())

def test_Array3D():
    minind=Int3BasicCoordinate(3)
    maxind=Int3BasicCoordinate(9)
    maxind[3]=11;
    indrange=IndexRange3D(minind,maxind)
    a3=FloatArray3D(indrange)
    minindtest=Int3BasicCoordinate(1)
    maxindtest=Int3BasicCoordinate(1)
    a3.get_regular_range(minindtest, maxindtest)
    assert minindtest==minind
    assert maxindtest==maxind
    assert a3.shape()==(7,7,9)
    a3.fill(2)
    ind=Int3BasicCoordinate((4,5,6))
    assert a3[ind]==2
    a3[ind]=9
    assert a3[(4,5,6)]==9
    assert a3.find_max()==9

def test_FloatVoxelsOnCartesianGrid():
    origin=FloatCartesianCoordinate3D(0,1,6)
    gridspacing=FloatCartesianCoordinate3D(1,1,2)
    minind=Int3BasicCoordinate(3)
    maxind=Int3BasicCoordinate(9)
    indrange=IndexRange3D(minind,maxind)
    image=FloatVoxelsOnCartesianGrid(indrange, origin,gridspacing)
    org= image.get_origin()
    assert org==origin
    image.fill(2)
    ind=Int3BasicCoordinate((4,4,4))
    assert image[ind]==2
    assert image[(5,3,4)]==2
    # construct from array
    a3=FloatArray3D(indrange)
    a3.fill(1.4);
    image=FloatVoxelsOnCartesianGrid(a3, origin,gridspacing)
    assert abs(image[ind]-1.4)<.001
    # change original array
    a3.fill(2)
    # shouldn't change image constructed from array
    assert abs(image[ind]-1.4)<.001

def test_zoom_image():
    # create test image
    origin=FloatCartesianCoordinate3D(3,1,6)
    gridspacing=FloatCartesianCoordinate3D(1,1,2)
    minind=Int3BasicCoordinate((0,-9,-9))
    maxind=Int3BasicCoordinate(9)
    indrange=IndexRange3D(minind,maxind)
    image=FloatVoxelsOnCartesianGrid(indrange, origin,gridspacing)
    image.fill(1)
    # find coordinate of middle of image for later use (independent of image sizes etc)
    [min_in_mm, max_in_mm]=stirextra.get_physical_coordinates_for_bounding_box(image)
    try:
        middle_in_mm=FloatCartesianCoordinate3D((min_in_mm+max_in_mm)/2.)
    except:
        # SWIG versions pre 3.0.11 had a bug, which we try to work around here
        middle_in_mm=FloatCartesianCoordinate3D((min_in_mm+max_in_mm).__div__(2))

    # test that we throw an exception if ZoomOptions is out-of-range
    try:
        zo=ZoomOptions(42)
        assert False
    except:
        assert True

    zoom=2
    offset=1
    new_size=6
    zoomed_image=zoom_image(image, zoom, offset, offset, new_size)
    ind=zoomed_image.get_indices_closest_to_physical_coordinates(middle_in_mm)
    assert zoomed_image[ind]==1./(zoom*zoom)
    # awkward syntax...
    zoomed_image=zoom_image(image, zoom, offset, offset, new_size, ZoomOptions(ZoomOptions.preserve_sum))
    assert abs(zoomed_image[ind]-1./(zoom*zoom))<.001
    zoomed_image=zoom_image(image, zoom, offset, offset, new_size, ZoomOptions(ZoomOptions.preserve_values))
    assert abs(zoomed_image[ind]-1)<.001
    zoomed_image=zoom_image(image, zoom, offset, offset, new_size, ZoomOptions(ZoomOptions.preserve_projections))
    assert abs(zoomed_image[ind]-1./(zoom))<.001
    
def test_Scanner():
    s=Scanner.get_scanner_from_name("ECAT 962")
    assert s.get_num_rings()==32
    assert s.get_num_detectors_per_ring()==576
    #l=s.get_all_names()
    #print s
    # does not work
    #for a in l:
    #    print a

def test_Bin():
    segment_num=1;
    view_num=2;
    axial_pos_num=3;
    tangential_pos_num=4;
    bin=Bin(segment_num, view_num, axial_pos_num, tangential_pos_num);
    assert bin.bin_value==0;
    assert bin.segment_num==segment_num;
    assert bin.view_num==view_num;
    assert bin.axial_pos_num==axial_pos_num;
    assert bin.tangential_pos_num==tangential_pos_num;
    bin.segment_num=5;
    assert bin.segment_num==5;
    bin_value=0.3;
    bin.bin_value=bin_value;
    assert abs(bin.bin_value-bin_value)<.01;
    bin=Bin(segment_num, view_num, axial_pos_num, tangential_pos_num, bin_value);
    assert abs(bin.bin_value-bin_value)<.01;
    
def test_ProjDataInfo():
    s=Scanner.get_scanner_from_name("ECAT 962")
    #ProjDataInfoCTI(const shared_ptr<Scanner>& scanner_ptr,
    #		  const int span, const int max_delta,
    #             const int num_views, const int num_tangential_poss, 
    #
    projdatainfo=ProjDataInfo.ProjDataInfoCTI(s,3,9,8,6)
    #print projdatainfo
    assert projdatainfo.get_scanner().get_num_rings()==32
    sinogram=projdatainfo.get_empty_sinogram(1,2)
    assert sinogram.sum()==0
    assert sinogram.get_segment_num()==2
    assert sinogram.get_axial_pos_num()==1
    assert sinogram.get_num_views() == projdatainfo.get_num_views()
    assert sinogram.get_proj_data_info() == projdatainfo

def test_ProjData_from_to_Array3D():
    # define a projection with some dummy data (filled with segment no.)
    s=Scanner.get_scanner_from_name("ECAT 962")
    projdatainfo=ProjDataInfo.ProjDataInfoCTI(s,3,9,8,6)
    projdata=ProjDataInMemory(ExamInfo(),projdatainfo)
    for seg_idx in range(projdata.get_min_segment_num(),projdata.get_max_segment_num()+1):
        segment=projdata.get_empty_segment_by_sinogram(seg_idx)
        segment.fill(seg_idx)
        projdata.set_segment(segment)

    # Check we actually put the data in (not just zeros)
    assert all([all([x==s for x in projdata.get_segment_by_sinogram(s).flat()])
                for s in range(projdata.get_min_segment_num(),projdata.get_max_segment_num()+1)])

    # convert to Array3D and back again
    array3D=projdata.to_array()
    new_projdata=ProjDataInMemory(ExamInfo(),projdatainfo)
    new_projdata.fill(array3D.flat())

    # assert every data point is equal
    assert all(a==b for a, b in zip(projdata.to_array().flat(),new_projdata.to_array().flat()))

