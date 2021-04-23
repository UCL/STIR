% Demo of how to use STIR from matlab to reconstruct some data

% Copyright 2012-06-05 - 2013 Kris Thielemans
% Copyright 2014, 2015 - University College London
% This file is part of STIR.
%
% SPDX-License-Identifier: Apache-2.0
%
% See STIR/LICENSE.txt for details

%% go to directory with input files
cd ../recon_demo
%% initialise reconstruction object
% we will do this here via a .par file 
recon=stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM.par');
%% now modify a few settings from in MATLAB for illustration
recon.set_num_subsets(2);
% set filenames to save subset sensitivities (for illustration purposes)
poissonobj=recon.get_objective_function();
poissonobj.set_subsensitivity_filenames('sens_subset%d.hv');
poissonobj.set_recompute_sensitivity(true)

%% get initial image
target=stir.FloatVoxelsOnCartesianGrid.read_from_file('init.hv');
% we will just fill the whole array with 1 here
target.fill(1)

%% run a few iterations and plot intermediate results
s=recon.set_up(target);
if (isequal(s,stir.Succeeded(stir.Succeeded.yes)))
    figure()
    hold on;
    for iter=1:4
        fprintf( '\n--------------------- Subiteration %d\n', iter);
        recon.set_start_subiteration_num(iter)
        recon.set_num_subiterations(iter)
        s=recon.reconstruct(target);
        % currently we need to explicitly prevent recomputing sensitivity when we
        % call reconstruct() again in the next iteration
        poissonobj.set_recompute_sensitivity(false)
        % extract to matlab for plotting
        image=target.to_matlab();
        plot(image(:,30,10))
        drawnow
    end
    % plot slice of final image
    figure()
    imshow(image(:,:,10),[],'InitialMagnification','fit')
    drawnow
else
    fprintf ('Error setting-up reconstruction object\n')
end
