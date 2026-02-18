#%%
## Authors: Zekai Li (University Medical Center Groningen), Nicole Jurjew (University College London)
## Modified from Nicole Jurjew’s Vision preprocessing script for the Siemens Quadra PET/CT scanner (/STIR/examples/python/Vision_files_preprocess.py).
## This python file uses jupyter notebook style to allow execution in separate code blocks and to avoid RAM usage issues.

#%%
import os
import numpy as np
import sys
import re
import stir
import stirextra
import matplotlib.pyplot as plt

#%%
## You'll need to run the e7tools with the following commands
## to get smoothed randoms, 2D scatter and norm-sino out:
##
## in any batch-file, add the following 2 lines, then execute
##  set cmd= %cmd% -d ./Debug
##  set cmd= %cmd% --os scatter_520_2D.mhdr
##
## the e7tools provide the prompts sinogram in a compressed file-format.
## STIR can't read that, so you'll have to uncompress it first:
## VR20 may not always work for uncompressing, you need to try more versions (VG80).
## C:\Siemens\PET\bin.win64-VR20\intfcompr.exe -e path\to\compressed\sinogram\filename.mhdr --oe path\to\UNcompressed\sinogram\NEWfilename.mhdr

#%%
## Here's a list of files you need (from the e7tools)
## - prompts-sino_uncompr_00.s.hdr & .s
## - smoothed_randoms_00.h33 & .s (in the "Debug" folder)
## - norm3d_00.h33 & .a (in the "Debug" folder)
## - scatter_520_2D.s.hdr & .s (wherever you set the path to)
## - acf_00.h33 & .a (in the "Debug" folder)

#%%
##### if you have any Siemens image data that you want to read in, make sure:
##### change the word "image data" to "imagedata"
##### remove line "!image relative start time (sec)...""
##### remove line "!image duration (sec):=1200..."

#%%
##### Functions
def change_datafilename_in_interfile_header(new_header_filename, header_filename, data_filename):
    with open(header_filename) as f:
        data = f.read()
    poss = re.search(r'name of data file\s*:=[^\n]*', data).span()
    data = data.replace(data[poss[0]:poss[1]], \
        'name of data file:={}'.format(data_filename))
    with open(new_header_filename, 'w') as f2:
        f2.write(data)

def DOI_adaption(projdata, DOI_new):
    proj_info = projdata.get_proj_data_info()

    DOI = proj_info.get_scanner().get_average_depth_of_interaction()
    print('Current Depth of interaction:', DOI)
    proj_info.get_scanner().set_average_depth_of_interaction(DOI_new)
    DOI = proj_info.get_scanner().get_average_depth_of_interaction()
    print('New Depth of interaction:', DOI)

def check_if_compressed(header_filename):
    with open(header_filename) as f:
        data = f.read()
    try:
        match = re.search(r'compression\s*:=\s*(\w+)', data)
        if match.group(1) == 'off':
            print('Compression is off, can proceed')
        else:
            print('You are trying to read e7tools compressed data. Please uncompress first!')
            sys.exit()
    except Exception:
        print('No compression info found in header!')

def plot_2d_image(idx,vol,title,clims=None,cmap="viridis"):
    """Customized version of subplot to plot 2D image"""
    plt.subplot(*idx)
    plt.imshow(vol,cmap=cmap)
    if clims is not None:
        plt.clim(clims)
    plt.colorbar(shrink=.5, aspect=.9)
    plt.title(title)
    plt.axis("off")

def change_datatype_in_interfile_header(header_name, data_type, num_bytes_per_pixel):
    with open(header_name) as f:
        data = f.read()
    poss = re.search(r'number format\s*:=[^\n]*', data).span()
    data = data.replace(data[poss[0]:poss[1]], \
        '!number format:={}'.format(data_type))
    poss = re.search(r'!number of bytes per pixel\s*:=[^\n]*', data).span()
    data = data.replace(data[poss[0]:poss[1]], \
        '!number of bytes per pixel:={}'.format(num_bytes_per_pixel))
    with open(header_name, 'w') as f2:
        f2.write(data)

def remove_scan_data_lines_from_interfile_header(header_filename_new, header_filename_old):
    with open(header_filename_old) as f:
        data = f.read()

    data_type_string = r'scan data type description[^\n]*\s*:=\s*[^\n]*\n'
    data = re.sub(data_type_string, '', data)

    num_data_types_string = r'number of scan data types[^\n]*\s*:=\s*[^\n]*\n'
    data = re.sub(num_data_types_string, '', data)

    with open(header_filename_new, 'w') as f2:
        f2.write(data)

def remove_IMGDATADESC_lines_from_interfile_header(header_filename_new, header_filename_old):
    with open(header_filename_old) as f:
        data = f.read()

    pattern = r'!IMAGE DATA DESCRIPTION:=.*'
    data = re.sub(pattern, '', data, flags=re.DOTALL)

    with open(header_filename_new, 'w') as f2:
        f2.write(data)

def remove_data_offset(header_filename_new, header_filename_old):
    with open(header_filename_old) as f:
        data = f.read()

    data_type_string = r'data offset in bytes[^\n]*\s*:=\s*[^\n]*\n'
    data = re.sub(data_type_string, '', data)

    with open(header_filename_new, 'w') as f2:
        f2.write(data)

def add_data_offset(header_filename_new, header_filename_old):
    with open(header_filename_old) as f:
        data = f.read()

    offset_string = '\ndata offset in bytes[1]:= 1202136000'
    pattern = r'(%TOF mashing factor\s*:=[^\n]*)'
    data = re.sub(pattern, r'\1' + offset_string, data)

    with open(header_filename_new, 'w') as f2:
        f2.write(data)

def replace_siemens_convention_in_interfile_header(header_name_new, header_name):
    with open(header_name) as f:
        data = f.read()

    poss = re.search(r'matrix axis label\[2\]:=plane', data).span()
    data = data.replace(data[poss[0]:poss[1]], \
        'matrix axis label[2]:=sinogram views')

    poss = re.search(r'matrix axis label\[3\]:=projection', data).span()
    data = data.replace(data[poss[0]:poss[1]], \
        'matrix axis label[3]:=number of sinograms')

    with open(header_name_new, 'w') as f2:
        f2.write(data)

def change_max_ring_distance(header_name_new, header_name, max_ring_diff):
    with open(header_name) as f:
        data = f.read()

    poss = re.search(r'%maximum ring difference\s*:=[^\n]*', data).span()
    data = data.replace(data[poss[0]:poss[1]], \
        '%maximum ring difference:={}'.format(int(max_ring_diff)))

    with open(header_name_new, 'w') as f2:
        f2.write(data)

def remove_tof_dimension(header_name_new, header_name):
    with open(header_name) as f:
        data = f.read()

    poss = re.search(r'number of dimensions\s*:=[^\n]*', data).span()
    data = data.replace(data[poss[0]:poss[1]], \
        'number of dimensions:={}'.format(3))

    data_type_string = r'matrix size\[4\]\s*:=[^\n]*\n'
    data = re.sub(data_type_string, '', data)

    data_type_string = r'matrix axis label\[4\]*\s*:=TOF bin*\n'
    data = re.sub(data_type_string, '', data)

    data_type_string = r'scale factor \(ps\/bin\) (.*)\n'
    data = re.sub(data_type_string, '', data)

    data_type_string = r'%TOF mashing factor[^\n]*\s*:=\s*[^\n]*\n'
    data = re.sub(data_type_string, '%TOF mashing factor :=0\n', data)

    data_type_string = r'%number of TOF time bins[^\n]*\s*:=\s*[^\n]*\n'
    data = re.sub(data_type_string, '', data)

    with open(header_name_new, 'w') as f2:
        f2.write(data)


#%%
data_folder_PATH = r'/PATHtosinograms'

###################### DOI ADAPTION ############################
## after comparing e7tools and STIR forward projections, we've found out we have to change
## the crystal depth of interaction (DOI) from 7mm to 10mm to minimize the differences.
# if apply_DOI_adaption: DOI_adaption(prompts_from_e7, 10), then save the file
apply_DOI_adaption = False

####### existing files #######
prompts_header_filename = 'uncompressed_prompts.s.hdr'
randoms_data_filename = 'smoothed_rand_00.s'
scatter_2D_header_filename = 'scatter_520_2D_00_00.s.hdr'
norm_sino_data_filename= 'norm3d_00.a'
attenuation_corr_factor_data_filename = 'acf_00.a'
STIR_output_folder = 'processing'

####### variables to chose #######
# header-name for prompts as we don't want to overwrite the Siemens header
prompts_header_to_read_withSTIR = prompts_header_filename[:-6] + '_readwSTIR.s.hdr'
# STIR writes the (DOI-adapted) prompts out to a STIR-file:
prompts_filename_STIR_corr_DOI = 'prompts.hs'
# header-name for attenuation correction factors as we don't want to overwrite the Siemens header
attenuation_corr_factor_header_to_read_withSTIR = attenuation_corr_factor_data_filename[:-2] + '_readwSTIR.s.hdr'
# header-name for randoms as we don't want to overwrite the Siemens header
norm_sino_to_read_withSTIR = norm_sino_data_filename[:-2] + '_readwSTIR.s.hdr'
# STIR writes the DOI-adapted, negative corrected norm-sino to
norm_filename_fSTIR = 'norm_sino_fSTIR.hs'
# header-name for randoms as we don't want to overwrite the Siemens header
randoms_header_to_read_withSTIR = randoms_data_filename[:-2] + '_readwSTIR.s.hdr'
# header-name for randoms as we don't want to overwrite the Siemens header
scatter_2D_header_to_read_withSTIR = scatter_2D_header_filename[:-6] + '_readwSTIR.s.hdr'
# STIR writes the (DOI-adapted) randoms to
randoms_adapted_DOI_filename = randoms_data_filename[:-2] + '_DOI_fSTIR.hs'
# STIR writes the (DOI-adapted), inverse SSRB, unnormalized scatter to
scatter_3D_norm_filename = 'scatter_3D_normalized.hs'
# STIR writes the additive term (that's normalized scatter + normalized randoms, attenuation corrected) to:
additive_term_filename_fSTIR = 'additive_term.hs'
# STIR writes the multiplicative term (that's norm_sino * attenuation_CORRECTION_factors) to:
multi_term_filename_fSTIR = 'mult_factors_forSTIR.hs'

#%%
os.chdir(data_folder_PATH)

try:
    os.mkdir(STIR_output_folder)
except FileExistsError:
    print("STIR output folder exists, files are overwritten")

#%%
###################### PROMPTS FILE ############################
##### first, we check if the prompts file is compressed
check_if_compressed(prompts_header_filename)
##### as the e7tools run on Windows, the data file-name needs to be changed
change_datafilename_in_interfile_header(prompts_header_to_read_withSTIR, prompts_header_filename ,prompts_header_filename[:-4])
# ## we're ready to read the prompts with STIR now
prompts_from_e7 = stir.ProjData.read_from_file(prompts_header_to_read_withSTIR)
# Directly read as numpy array
prompts_arr = stirextra.to_numpy(prompts_from_e7)
# Write on the disk
proj_info = prompts_from_e7.get_proj_data_info()

#%%
prompts_from_e7=stir.ProjDataInterfile(prompts_from_e7.get_exam_info(), proj_info, os.path.join(STIR_output_folder, prompts_filename_STIR_corr_DOI))
prompts_from_e7.fill(prompts_arr.flat)

#%%
# Variables for checking sinograms
central_slice = proj_info.get_num_axial_poss(0)//2
TOF_bin = proj_info.get_num_tof_poss()//2
view = proj_info.get_num_views()//2
# to draw line-profiles, we'll average over a few slices, specify how many:
thickness_half = 5

#%%
# Sanity check for knowing the total counts
print(np.sum(prompts_arr))

# %%
###################### NORMALIZATION ############################
### we're using the prompts-header as a template for the norm-header
change_datafilename_in_interfile_header(norm_sino_to_read_withSTIR, prompts_header_filename, norm_sino_data_filename)
change_datatype_in_interfile_header(norm_sino_to_read_withSTIR, 'float', 4)

#%%
#### ready to read in norm-sino with STIR
norm_sino = stir.ProjData.read_from_file(norm_sino_to_read_withSTIR)
###################### DOI ADAPTION ############################
## after comparing e7tools and STIR forward projections, we've found out we have to change
## the crystal depth of interaction (DOI) from 7mm to 10mm to minimize the differences.
if apply_DOI_adaption: DOI_adaption(norm_sino, 10)
norm_sino_arr = stirextra.to_numpy(norm_sino)

##### In case there were bad miniblocks during your measurement, the norm-file
##### might contain negative values. We'll set them to a very high value here, such
##### that the detection efficiencies (1/norm-value) will be 0 (numerically)
norm_sino_arr[norm_sino_arr<=0.] = 10^37
#### this is the data STIR needs in an Acquisition Sensitivity model, so we'll write it out
#%%
norm_sino_STIR = stir.ProjDataInterfile(prompts_from_e7.get_exam_info(), proj_info, os.path.join(STIR_output_folder,norm_filename_fSTIR))
norm_sino_STIR.fill(norm_sino_arr.flat)
#%%
for i in range(33):
    plt.figure()
    plot_2d_image([1,1,1],norm_sino_arr[i, central_slice,:,:],'norm TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'norm_sino.png'), transparent=False, facecolor='w')
    plt.show()
    
#%%
det_eff_arr = np.nan_to_num(np.divide(1, norm_sino_arr))

#%%
for i in range(33):
    plt.figure()
    plot_2d_image([1,1,1],det_eff_arr[i, central_slice,:,:],'det. eff., TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'det_effs.png'), transparent=False, facecolor='w')
    plt.show()

#%%
###################### RANDOMS ############################
### below, we will use the prompts-header as a template for a header to read in Siemens randoms
change_datafilename_in_interfile_header(randoms_header_to_read_withSTIR, prompts_header_filename, randoms_data_filename)
### need to change the file type, because prompts are in unsigned short, randoms are in float
change_datatype_in_interfile_header(randoms_header_to_read_withSTIR, 'float', 4)

### The data type descriptions are true for the prompts, but not
### for the randoms, so we need to remove them (and some other info) from the header.
remove_scan_data_lines_from_interfile_header(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)
remove_IMGDATADESC_lines_from_interfile_header(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)

### The first set of sinograms in the randoms file is the "delayeds", so we need
### to tell STIR to ignore that by setting a data offset. The other offset fields,
### we can just remove, in line with the "data types" we removed above.
remove_data_offset(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)
add_data_offset(randoms_header_to_read_withSTIR, randoms_header_to_read_withSTIR)

#%%
# #### read in again & plot to see if it worked
randoms = stir.ProjData.read_from_file(randoms_header_to_read_withSTIR)
if apply_DOI_adaption: DOI_adaption(randoms, 10)
randoms_arr = stirextra.to_numpy(randoms)

#%%
# Sanity check
print(np.sum(randoms_arr))

#%%
for i in range(33):
    plt.figure()
    plot_2d_image([1,1,1],randoms_arr[i, central_slice,:,:],'randoms, TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'randoms_sino.png'), transparent=False, facecolor='w')
    plt.show()

#%%
###################### SCATTER ############################
# alter e7-header of scatter manually to read 2D scatter
seg0_max_rd = 322 # TODO: get this via proj-data-info; currently cannot “downcast” to ProjDataInfoCylindrical
remove_data_offset(scatter_2D_header_to_read_withSTIR, scatter_2D_header_filename)
remove_scan_data_lines_from_interfile_header(scatter_2D_header_to_read_withSTIR, scatter_2D_header_to_read_withSTIR)
replace_siemens_convention_in_interfile_header(scatter_2D_header_to_read_withSTIR, scatter_2D_header_to_read_withSTIR)
change_max_ring_distance(scatter_2D_header_to_read_withSTIR, scatter_2D_header_to_read_withSTIR, seg0_max_rd)

#%%
## read in 2D scatter with STIR
scatter_2D_normalized = stir.ProjData.read_from_file(scatter_2D_header_to_read_withSTIR)
if apply_DOI_adaption: DOI_adaption(scatter_2D_normalized, 10)

#%%
## we hand the object of the final 3D sinogram over to the inverse SSRB function.
scatter_3D_normalized = stir.ProjDataInterfile(prompts_from_e7.get_exam_info(), proj_info, os.path.join(STIR_output_folder,scatter_3D_norm_filename))
stir.inverse_SSRB(scatter_3D_normalized, scatter_2D_normalized)
#%%
# plot to see if it worked
scatter_3D_normalized = stir.ProjData.read_from_file(os.path.join(STIR_output_folder,scatter_3D_norm_filename))
scatter_3D_norm_arr = stirextra.to_numpy(scatter_3D_normalized)
#%%
# Sanity check, however the e7 tools only output the 2D scatter, the 3D scatter here is not exactly the same as their
print(np.sum(scatter_3D_norm_arr))
#%%
for i in range(33):
    plt.figure()
    plot_2d_image([1,1,1],scatter_3D_norm_arr[i, central_slice,:,:],'scatter, normalized, TOF bin {}'.format(i))
    plt.show()

#%%
###################### ADDITIVE TERM ############################
#### the additive term is what is added to the projected estimate, before any multiplicative factors
#### are applied. Therefore, we need to normalize randoms and add (normalized) scatter to it.
randoms_arr_normalised = randoms_arr * norm_sino_arr
#%%
add_sino_arr_normalised = randoms_arr_normalised + scatter_3D_norm_arr

#%%
###################### ATTENUATION CORRECTION FACTORS ############################
# to get the correct additive term, we need to apply attenuation correction.
# we'll use the ones from the e7tools here.
change_datafilename_in_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, prompts_header_filename, attenuation_corr_factor_data_filename)
change_datatype_in_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, 'float', 4)
remove_scan_data_lines_from_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)
remove_IMGDATADESC_lines_from_interfile_header(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)
remove_data_offset(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)
remove_tof_dimension(attenuation_corr_factor_header_to_read_withSTIR, attenuation_corr_factor_header_to_read_withSTIR)

#%%
acf_sino_nonTOF = stir.ProjData.read_from_file(attenuation_corr_factor_header_to_read_withSTIR)
if apply_DOI_adaption: DOI_adaption(acf_sino_nonTOF, 10)

#%%
#### expand to TOF as normalization data is TOF
ai_arr = stirextra.to_numpy(acf_sino_nonTOF)
acf_arr = np.repeat(ai_arr, 33, axis=0)

#%%
###################### ATTENUATION FACTORS ############################
afs = np.divide(1,acf_arr)

#%%
for i in range(33):
    plt.figure()
    plot_2d_image([1,1,1],acf_arr[i, central_slice,:,:],'acfs, TOF bin {}'.format(i))
    plt.show()

#%%
add_sino_arr = add_sino_arr_normalised * acf_arr

#%%
add_sino = stir.ProjDataInterfile(prompts_from_e7.get_exam_info(), proj_info, os.path.join(STIR_output_folder,additive_term_filename_fSTIR))
add_sino.fill(add_sino_arr.flat)

#%%
for i in range(33):
    plt.figure()
    plot_2d_image([1,1,1],add_sino_arr[i, central_slice,:,:],'additive term, TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'RANDOMS.png'), transparent=False, facecolor='w')
    plt.show()

#%%
######### now let's write out the multiplicative factors f. STIR
multi_factors_STIR_arr = norm_sino_arr * acf_arr

#%%
for i in range(33):
    plt.figure()
    plot_2d_image([1,1,1],multi_factors_STIR_arr[i, central_slice,:,:],'multi_factors_STIR_arr, TOF bin {}'.format(i))
    if i == TOF_bin: plt.savefig(os.path.join(STIR_output_folder,'multi_factors_STIR_arr.png'), transparent=False, facecolor='w')
    plt.show()
    
#%%
multi_factors_STIR = stir.ProjDataInterfile(prompts_from_e7.get_exam_info(), proj_info, os.path.join(STIR_output_folder,multi_term_filename_fSTIR))
multi_factors_STIR.fill(multi_factors_STIR_arr.flat)

#%%
##### to compare the additive term with the acquisition data, we need to
##### pre-correct the prompts with the multiplicative factors
prompts_precorr = prompts_arr * acf_arr * norm_sino_arr

#%%
#### PLOT ADDITIVE TERM
#### draw line-profiles to check if all's correct
TOF_bin = 6

fig, ax = plt.subplots(figsize = (8,6))

ax.plot(np.mean(prompts_precorr[TOF_bin, central_slice-thickness_half:central_slice+thickness_half, 0, :], axis=(0)), label='Prompts, corr-multfactors')
ax.plot(np.mean(add_sino_arr[TOF_bin, central_slice-thickness_half:central_slice+thickness_half, 0, :], axis=(0)), label='additive term')

ax.set_xlabel('Radial distance (bin)')
ax.set_ylabel('total counts')
ax.set_title('TOF bin:' + str(TOF_bin))
ax.legend()
plt.tight_layout()
plt.suptitle('Lineprofiles - Avg over 10 central slices')
plt.savefig(os.path.join(STIR_output_folder,'additive_term_lineprofiles.png'), transparent=False, facecolor='w')
plt.tight_layout()