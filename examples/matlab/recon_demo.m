% Demo of how to use STIR from matlab to reconstruct some data

% Copyright 2012-06-05 - 2013 Kris Thielemans
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
recon=stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM.par')
%% now modify a few settings from in Python for illustration
recon.set_num_subsets(2);
poissonobj=recon.get_objective_function()
poissonobj.set_sensitivity_filename('sens.hv');
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
        poissonobj.set_recompute_sensitivity(false)
        image=target.to_matlab();
        plot(image(:,30,10))
        drawnow
    end
    % plot slice of final image
    figure()
    imshow(image(:,:,10),[],'InitialMagnification','fit')
    drawnow
else
    fprintf ('Error setting-up reconstruction object')
end
