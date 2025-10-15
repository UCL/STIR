# %%
import stir
import stirextra
import matplotlib.pyplot as plt
import os

# go to directory with input files
os.chdir('../recon_demo')

# %%
# initialise reconstruction object
# we will do this here via a .par file
recon = stir.OSMAPOSLReconstruction3DFloat('recon_demo_OSEM.par')
# now modify a few settings from in Python for illustration
recon.set_num_subsets(2)
num_subiterations = 4
# set filenames to save subset sensitivities (for illustration purposes)
poissonobj = recon.get_objective_function()
poissonobj.set_subsensitivity_filenames('sens_subset%d.hv')
poissonobj.set_recompute_sensitivity(True)


# %%
# construct image related to the data to reconstruct
projdata=stir.ProjData.read_from_file('smalllong.hs');
# use smaller voxels than the default
zoom=2.216842;
target=stir.FloatVoxelsOnCartesianGrid(projdata.get_proj_data_info(), zoom);
# get initial image
help(target)
# target = stir.FloatVoxelsOnCartesianGrid.read_from_file('init.hv')
# we will just fill the whole array with 1 here
target.fill(1)
s = recon.set_up(target)

# %%

# compute gradient of objective function
# create a copy to store the gradient
gradient=target.get_empty_copy();
# compute gradient
subset_num=1;
poissonobj.compute_sub_gradient(gradient,target,subset_num)

# extract to python for plotting
npimage = stirextra.to_numpy(gradient)
plt.plot(npimage[10, 30, :])
plt.show()

# this is useful to find the EM update (i.e. multiply with image)
poissonobj.compute_sub_gradient_without_penalty_plus_sensitivity(gradient,target,subset_num)
# extract to python for plotting
npimage = stirextra.to_numpy(gradient)
plt.plot(npimage[10, 30, :])
plt.show()

# The followin is just the first iteration you need to create a for
# loop to run all the iterations and subsets
EMupdate = target*gradient
# extract to python for plotting
npimage = stirextra.to_numpy(EMupdate)
plt.plot(npimage[10, 30, :])
plt.show()



