% Demo of how to use STIR from matlab to compute gradients of the objective function

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
%% initialise reconstruction object
% we will do this here via a .par file 
recon=stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM.par');
%% now modify a few settings from in MATLAB for illustration
recon.set_num_subsets(2);
poissonobj=recon.get_objective_function();
poissonobj.set_subsensitivity_filenames('sens_subset%d.hv')
poissonobj.set_recompute_sensitivity(true)
%% construct image related to the data to reconstruct
projdata=stir.ProjData.read_from_file('smalllong.hs');
% use smaller voxels than the default
zoom=2.216842;
target=stir.FloatVoxelsOnCartesianGrid(projdata.get_proj_data_info(), zoom);
%% set-up reconstruction object
% this will already compute the (subset) sensitivities
s=recon.set_up(target);
%% compute gradient of objective function 
% put some data in the image
target.fill(1);
% create a copy to store the gradient
gradient=target.get_empty_copy();
% compute gradient
subset_num=1;
poissonobj.compute_sub_gradient(gradient,target,subset_num)
%%
%% display
gradientmatlab=gradient.to_matlab();
figure;imshow(gradientmatlab(:,:,10),[])
%% compute without sensitivity term (and any penalty)
% this would be useful to find the EM update (i.e. multiply with image)
poissonobj.compute_sub_gradient_without_penalty_plus_sensitivity(gradient,target,subset_num)
%%EM update
% This doesn't work yet (mtimes not implemented in matlab via SWIG yet)
% EMupdate=gradient*target;
EMupdatematlab=target.to_matlab() .* gradient.to_matlab();
figure;imshow(EMupdatematlab(:,:,10),[])
