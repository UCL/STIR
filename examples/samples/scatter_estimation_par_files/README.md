# Example files for running scatter estimation

Files made by Nikos Efthimou and fine-tuned by Kris Thielemans.<br>
Copyright University of Hull 2018-2019<br>
copyright University College London 2016, 2020<br>
Distributed under the Apache 2.0 License


This example is set-up using environment variables such that you can use these
files without modifying them if you want. Of course, you can also edit the
files to use explicit values instead.

The main file is `scatter_estimation.par`. It points to the other files using
the `scatter_pardir` environment variable.

Usage would be something like this
```sh
## set location of files (adjust for your location)
scatter_pardir=~/devel/STIR/examples/samples/scatter_estimation_par_files
## input files
sino_input=myfile.hs
atnimg=myattenuationimage.hv
NORM=normfactors.hs
acf3d=acf.hs
randoms3d=randoms.hs
## scatter settings
num_scat_iters=5 # this is the default value
## recon settings during scatter estimation
# adjust for your scanner (needs to divide number of views/4 as usual)
scatter_recon_num_subsets=7
# keep num_scatter_iters*scatter_recon_num_subiterations relatively small as everything is at low resolution
scatter_recon_num_subiterations=7
## filenames for output
mask_projdata_filename=mask.hs
mask_image=mask_image.hv
scatter_prefix=scatter
total_additive_prefix=addsino

export scatter_pardir
export sino_input atnimg NORM acf3d randoms3d
export scatter_recon_num_subsets scatter_recon_num_subiterations
export mask_projdata_filename mask_image scatter_prefix total_additive_prefix

estimate_scatter $scatter_pardir/scatter_estimation.par
```
The last files written with `${total_additive_prefix}_#.hs` can be used as an
`additive sinogram` in further reconstructions.

See a full example in the `examples/Siemens-mMR` folder, and recon_test_pack/run_scatter_test.sh.
