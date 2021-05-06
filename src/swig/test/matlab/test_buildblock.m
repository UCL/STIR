%% Test file for STIR building blocks from MATLAB

%    Copyright (C) 2013-2015 University College London
%    This file is part of STIR.
%
%    SPDX-License-Identifier: Apache-2.0
%
%    See STIR/LICENSE.txt for details
%% Instructions
% You can just run this file (probably after adding the stir files to your
% matlab path)
%addpath('C:\Users\kris\Documents\devel\UCL-STIR\src\swig')
%addpath('~/devel/build/UCL-STIR/gcc-4.8.1-debug/src/swig')
% if there's no assertion errors, everything is fine
%%
fprintf(['You will see 4 warnings about arrays and incompatible sizes, even if the test works.\n'...
    'This is a defect of current SWIG. Ignore.\n']);
%% basic tests on coordinates
c=stir.make_FloatCoordinate(3.3,4.4,5);
assert(abs(c(1)-3.3)<1e-5)
assert(c(3)==5)
cm=c.to_matlab();
assert(cm(3)==5);
c.fill(6)
assert(c(2)==6)
c.fill([1;2;3])
assert(c(1)==1)

%% basic tests on 1D arrays
a=stir.FloatArray1D(3,4);
tmp=a.get_length();
assert(tmp==2)
b=stir.IndexRange1D(2,6);
b.resize(2,6)
tmp=b.get_min_index();
assert(tmp==2)
tmp=b.get_max_index();
assert(tmp==6)
c=b;
tmp=c.get_max_index();
assert(tmp==6)
a=stir.FloatArray1D(3,4);
a.fill(5)
tmp=a.find_min();
assert(tmp==5)
%%
% tests filling with vectors
a.fill([2; 3])
assert(a(3)==2)
assert(a(4)==3)
try
  a.fill([2; 3;4])
  assert(false, 'fill with wrong size should have failed')
catch e
end
%% wrong dimensions, should fail
try 
  a.fill(ones(1,2))
  assert(false, 'fill with wrong dimension should have failed')
catch e
end
% test convert to array
tmp=[2; 3];
a.fill(tmp+1)
tmp2=a.to_matlab();
assert(isequal(tmp2,tmp+1))

%% basic tests on 2D arrays
% next is currently not wrapped
% r=stir.IndexRange2D(1,4,2,7)
c=stir.make_IntCoordinate(3,6);
assert(isa(c,'stir.Int2BasicCoordinate'))
b=stir.IndexRange2D(c);
a=stir.FloatArray2D(b);
tmp=stir.FloatArray2D(a);
assert(tmp.get_num_dimensions() ==2)
tmp.fill(4);
assert(isa(tmp,'stir.FloatArray2D'))
assert(tmp.find_max()==4);
% original shouldn't be modified
assert(a.find_max()==0);
%% test conversion of arrays in 2D
tmp=[1 2; 3 4; 5 6];
a=stir.FloatArray2D(tmp);
% note complicated correspondence if indexing (reverse-order and 0-based vs 1-based)
assert(a(stir.make_IntCoordinate(1,2)) == tmp(3,2))
% test filling
a.fill(tmp+1)
assert(a(stir.make_IntCoordinate(1,2)) == tmp(3,2)+1)
tmp2=a.to_matlab();
assert(isequal(tmp2,tmp+1))
% test fill with wrong size
try
  a.fill(ones([2,3]))
  assert(false, 'fill with wrong size should have failed')
catch e
end
% wrong dimensions, should fail
try 
  a.fill(ones(2,3,3))
  assert(false, 'fill with wrong dimension should have failed')
catch e
end
% trivial upper dimension
c=stir.make_IntCoordinate(1,3);
b=stir.IndexRange2D(c);
a=stir.FloatArray2D(b);
a.fill([1;2;3]);
assert(a(stir.make_IntCoordinate(0,1))==2)

%% test conversion of arrays in 3D
tmp=[1 2 3 4; 5 6 7 8; 9 10 11 12];
tmp(:,:,2)=[1 2 3 4; 5 6 7 8; 9 10 11 12]+100;
a=stir.FloatArray3D(tmp);
assert(a(stir.make_IntCoordinate(1,3,2)) == tmp(3,4,2))
assert(isequal(tmp, a.to_matlab()));
% check if it's possible to create a FloatArray3D from a 2D matlab array
% (i.e. single slice)
tmp=[1 2 3 4; 5 6 7 8; 9 10 11 12];
a=stir.FloatArray3D(tmp);
assert(a(stir.make_IntCoordinate(0,1,2)) == tmp(3,2))
assert(isequal(tmp, a.to_matlab()));
%% basic tests on FloatVoxelsOnCartesianGrid
origin=stir.FloatCartesianCoordinate3D(0,1,6);
gridspacing=stir.FloatCartesianCoordinate3D(1,1,2);
minind=stir.Int3BasicCoordinate(3);
maxind=stir.Int3BasicCoordinate(9);
indrange=stir.IndexRange3D(minind,maxind);
image=stir.FloatVoxelsOnCartesianGrid(indrange, origin,gridspacing);
org= image.get_origin();

image.fill(2);
ind=stir.make_IntCoordinate(3,4,5);
tmp=image(ind);
assert(tmp==2)
image.paren_asgn(ind,6);
tmp=image(ind);
assert(tmp==6)
%% simple test on Scanner
s=stir.Scanner.get_scanner_from_name('ECAT 962');
% alternative
%s=stir.Scanner(stir.Scanner.E962());
assert (s.get_num_rings()==32)
assert (s.get_num_detectors_per_ring()==576)
%% tests on ProjDataInfo
% this doesn't work (no conversion), but probably rightly so
% projdatainfo=stir.ProjDataInfoCylindricalNoArcCorr (stir.ProjDataInfo.ProjDataInfoCTI(s,3,3,8,6))
s=stir.Scanner(stir.Scanner.E962());
projdatainfo=stir.ProjDataInfo.ProjDataInfoCTI(s,3,9,8,6);
%print projdatainfo
assert( projdatainfo.get_scanner().get_num_rings()==32)
sinogram=projdatainfo.get_empty_sinogram(1,2);
assert( sinogram.sum()==0)
assert( sinogram.get_segment_num()==2)
assert( sinogram.get_axial_pos_num()==1)
assert( sinogram.get_num_views() == projdatainfo.get_num_views())
assert( isequal(sinogram.get_proj_data_info(), projdatainfo))
% create some empty objects
segment=projdatainfo.get_empty_segment_by_view(0);
assert(segment.find_max()==0)
segment.fill(2);
segment2=stir.FloatSegmentByView(projdatainfo,0);
% check conversion
a=stir.FloatArray3D(segment);
assert(a.find_max()==2)

%% ProjDataInMemory
success=stir.Succeeded(stir.Succeeded.yes());

s=stir.Scanner(stir.Scanner.E962());
proj_data_info=stir.ProjDataInfo.ProjDataInfoCTI(s,3,9,8,6);
proj_data=stir.ProjDataInMemory(stir.ExamInfo(), proj_data_info);
seg=proj_data.get_segment_by_sinogram(0);

assert(seg.find_min()==0);
seg.fill(4);
assert(seg.find_min()==4);
assert(isequal(proj_data.set_segment(seg), success));
seg2=proj_data.get_segment_by_sinogram(0);
assert(isequal(seg2.to_matlab(), seg.to_matlab()));
