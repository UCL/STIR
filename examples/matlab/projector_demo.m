% Demo of how to use STIR from matlab to project some data

% Copyright 2014-2015 - University College London
% This file is part of STIR.
%
% SPDX-License-Identifier: Apache-2.0
%
% See STIR/LICENSE.txt for details

%% go to directory with input files
cd ../recon_demo
%%
projdata=stir.ProjData.read_from_file('smalllong.hs');
% use smaller voxels than the default
zoom=2.216842;
target=stir.FloatVoxelsOnCartesianGrid(projdata.get_proj_data_info(), zoom);
%% initialise the projection matrix 
% (using ray-tracing here)
projmatrix=stir.ProjMatrixByBinUsingRayTracing();
projmatrix.set_up(projdata.get_proj_data_info(), target);
%% construct projectors
forwardprojector=stir.ForwardProjectorByBinUsingProjMatrixByBin(projmatrix);
backprojector=stir.BackProjectorByBinUsingProjMatrixByBin(projmatrix);

%% create projection data for output of forward projection
% We'll just create the data in memory here
projdataout=stir.ProjDataInMemory(projdata.get_exam_info(), projdata.get_proj_data_info());
% Note: we could write to file, but it is right now a bit complicated to open a
% file for read/write:
%  inout=bitor(uint32(stir.ios.ios_base_in()),bitor(uint32(stir.ios.trunc()),uint32(stir.ios.out())));
%  projdataout=stir.ProjDataInterfile(projdata.get_exam_info(), projdata.get_proj_data_info(), 'stir_matlab_test.hs',inout);
%% forward project an image
target.fill(2);
forwardprojector.set_up(projdataout.get_proj_data_info(), target);
forwardprojector.forward_project(projdataout, target);
%% display
seg=projdataout.get_segment_by_sinogram(0);
segmatlab=seg.to_matlab();
figure;imshow(segmatlab(:,:,10)',[])
%% backproject original data
backprojector.set_up(projdataout.get_proj_data_info(), target);
backprojector.back_project(target, projdata);
%% display
targetmatlab=target.to_matlab();
figure;imshow(targetmatlab(:,:,10),[])
