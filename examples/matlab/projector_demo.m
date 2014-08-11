% Demo of how to use STIR from matlab to project some data

% Copyright 2014 - University College London
% This file is part of STIR.
%
% This file is free software; you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation; either version 2.1 of the License, or
% (at your option) any later version.
%
% This file is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
%
% See STIR/LICENSE.txt for details

%% go to directory with input files
cd ../recon_demo
%%
projdata=stir.ProjData.read_from_file('smalllong.hs');
% use smaller voxels than the default
zoom=2.216842;
target=stir.FloatVoxelsOnCartesianGrid(projdata.get_proj_data_info_ptr(), zoom);
%% initialise the projection matrix 
% (using ray-tracing here)
projmatrix=stir.ProjMatrixByBinUsingRayTracing();
projmatrix.set_up(projdata.get_proj_data_info_ptr(), target);
%% construct projectors
forwardprojector=stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix);
backprojector=stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix);

%% create projection data for output of forward projection
% we will currently write to file as at present we cannot read/write for some reason (TODO)
projdataout=stir.ProjDataInterfile(projdata.get_exam_info_ptr(), projdata.get_proj_data_info_ptr(), 'stir_matlab_test.hs');%,stir.ios_inout());
%projdataout=stir.ProjDataInMemory(examinfo, projdata.get_proj_data_info_ptr());
%% forward project an image
target.fill(2);
forwardprojector.forward_project(projdataout, target);
%% work-around read/write problem
% currently need to close and re-open for reading
clear projdataout
projdataout=stir.ProjData.read_from_file('stir_matlab_test.hs');
%% display
seg=projdataout.get_segment_by_sinogram(0);
segmatlab=seg.to_matlab();
figure;imshow(segmatlab(:,:,10)',[])
%% backproject original data
backprojector.back_project(target, projdata);
%% display
targetmatlab=target.to_matlab();
figure;imshow(targetmatlab(:,:,10),[])
