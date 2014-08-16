%% Test file for STIR IO from MATLAB

%    Copyright (C) 2013-2014 University College London
%    This file is part of STIR.
%
%    This file is free software; you can redistribute it and/or modify
%    it under the terms of the GNU Lesser General Public License as published by
%    the Free Software Foundation; either version 2.1 of the License, or
%    (at your option) any later version.

%    This file is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU Lesser General Public License for more details.
%
%    See STIR/LICENSE.txt for details
%% Instructions
% You can just run this file (probably after adding the stir files to your
% matlab path)
%addpath('C:\Users\kris\Documents\devel\UCL-STIR\src\swig')
% if there's no assertion errors, everything is fine
%% import all of stir into the current "namespace"
import stir.*
%%
%def test_FloatVoxelsOnCartesianGrid(tmpdir):
    %tmpdir.chdir()
    fprintf('Creating files in %s\n', pwd())
    origin=FloatCartesianCoordinate3D(0,1,6);
    gridspacing=FloatCartesianCoordinate3D(1,1,2);
    minind=Int3BasicCoordinate(3);
    maxind=Int3BasicCoordinate(9);
    indrange=IndexRange3D(minind,maxind);
    image=FloatVoxelsOnCartesianGrid(indrange, origin,gridspacing);
    ind=make_IntCoordinate(5,3,4);
    image.setel(ind,2);
    %print image.shape()
    output_format=InterfileOutputFileFormat();
    output_format.write_to_file('stir_matlab_test.hv', image) ;
    image2=FloatVoxelsOnCartesianGrid.read_from_file('stir_matlab_test.hv');
    assert (isequal(image.get_voxel_size(),image2.get_voxel_size()))
    %assert (image.shape()==image2.shape())
    %assert (get_physical_coordinates_for_bounding_box(image) == get_physical_coordinates_for_bounding_box(image2))
    assert(image2.getel(ind - image.get_min_indices() + image2.get_min_indices()) == image.getel(ind))
    assert(norm(reshape(image.to_matlab()-image2.to_matlab(),1,[]))<.1)
%%
%def test_ProjDataInfo(tmpdir):
%    tmpdir.chdir()
    fprintf('Creating files in %s\n', pwd())
    s=Scanner.get_scanner_from_name('ECAT 962');
    %ProjDataInfoCTI(const shared_ptr<Scanner>& scanner_ptr,
    %		  const int span, const int max_delta,
    %             const int num_views, const int num_tangential_poss, 
    %
    examinfo=ExamInfo();
    projdatainfo=ProjDataInfo.ProjDataInfoCTI(s,3,6,8,6);
    assert (projdatainfo.get_scanner().get_num_rings()==32)
    projdata=ProjDataInterfile(examinfo, projdatainfo, 'stir_matlab_test.hs');
    assert (projdata.get_min_segment_num()==-1)
    assert( projdata.get_max_segment_num()==+1)
    for seg=projdata.get_min_segment_num() : projdata.get_max_segment_num()
        segment=projdatainfo.get_empty_segment_by_view(seg);
        segment.fill(double(seg)+100.)
        assert(isequal(projdata.set_segment(segment),Succeeded(Succeeded.yes)))
    end
    % now delete object such that the file gets closed and we can read it
    delete( projdata)

    projdata2=ProjData.read_from_file('stir_matlab_test.hs');
    assert (isequal(projdatainfo,projdata2.get_proj_data_info()))
    for seg=projdatainfo.get_min_segment_num() : projdatainfo.get_max_segment_num()
        % construct same segment data as above (TODO: better to stick it into a list or so)
        segment=projdatainfo.get_empty_segment_by_view(seg);
        segment.fill(double(seg)+100);
        % read from file
        segment2=projdata2.get_segment_by_view(seg);
        % compare
        ind=stir.make_IntCoordinate(0,0,0);
        assert(segment.getel(ind) == segment2.getel(ind));
        assert(norm(reshape(segment.to_matlab()-segment2.to_matlab(),1,[]))<.1)
    end   
