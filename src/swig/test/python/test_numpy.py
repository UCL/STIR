# Test file for conversions from STIR to numpy arrays (using py.test)
# Use as follows:
# on command line
#     py.test test_numpy.py


#    Copyright (C) 2013, 2015 University College London
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

import py.test
from stir import *
import stirextra
# for Python2 and itertools.zip->zip (as in Python 3) 
try:
    import itertools.izip as zip
except ImportError:
    pass

def test_Array2D():
    minind=Int2BasicCoordinate((3,3));
    a=FloatArray2D(IndexRange2D(minind, Int2BasicCoordinate((9,8))))
    a.fill(2);
    ind=Int2BasicCoordinate((4,5));
    a[ind]=4
    np=stirextra.to_numpy(a);
    assert np[(0,0)]==2
    npind=(ind[1]-minind[1], ind[2]-minind[2])
    assert np[npind]==a[ind]
    np+=2
    a.fill(np.flat)
    assert np[(0,0)]==4
    
def test_Array2Diterator():
    a=FloatArray2D(IndexRange2D(Int2BasicCoordinate((1,3)), Int2BasicCoordinate((3,9))))
    for i1,i2 in zip(a.flat(), range(a.size_all())):
        i1=i2;
    np=stirextra.to_numpy(a);
    for i1,i2 in zip(a.flat(), np.flat):
        assert abs(i1-i2)<.01

def test_Array3D():
    minind=Int3BasicCoordinate((3,3,5));
    a=FloatArray3D(IndexRange3D(minind, Int3BasicCoordinate((9,8,7))))
    a.fill(2);
    ind=Int3BasicCoordinate((4,5,6));
    a[ind]=4
    np=stirextra.to_numpy(a);
    assert np[(0,0,1)]==2
    npind=(ind[1]-minind[1], ind[2]-minind[2], ind[3]-minind[3])
    assert np[npind]==a[ind]
    np+=2
    a.fill(np.flat)
    assert np[(0,0,1)]==4
    assert np[npind]==a[ind]

def test_Array3Diterator():
    a=FloatArray3D(IndexRange3D(Int3BasicCoordinate((1,3,-1)), Int3BasicCoordinate((3,9,5))))
    for i1,i2 in zip(a.flat(), range(a.size_all())):
        i1=i2;
    np=stirextra.to_numpy(a);
    for i1,i2 in zip(a.flat(), np.flat):
        assert abs(i1-i2)<.01

def test_ProjData():
    s=Scanner.get_scanner_from_name("ECAT 962")
    #ProjDataInfoCTI(const shared_ptr<Scanner>& scanner_ptr,
    #		  const int span, const int max_delta,
    #             const int num_views, const int num_tangential_poss, 
    #
    projdatainfo=ProjDataInfo.ProjDataInfoCTI(s,3,9,8,6)
    #print projdatainfo
    projdata=ProjDataInMemory(ExamInfo(), projdatainfo)
    np=stirextra.to_numpy(projdata)
    np+=2
    projdata.fill(np.flat)
    seg0=stirextra.to_numpy(projdata.get_segment_by_sinogram(0))
    assert(seg0.max() == 2)

