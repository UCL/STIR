//
//
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjectorByBinNiftyPETHelper

  \author Richard Brown


*/
/*
    Copyright (C) 2019, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

#include <fstream>
#include "stir/gpu/ProjectorByBinNiftyPETHelper.h"
#include <def.h>
#include <boost/format.hpp>
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

/// Read NiftyPET binary file
template <class dataType>
std::vector<dataType>
read_binary_file(std::string file_name)
{
    const char* stir_path = std::getenv("STIR_PATH");
    if (!stir_path)
        throw std::runtime_error("STIR_PATH not defined, cannot find data");

    std::string data_path = stir_path;
    data_path += "/examples/mMR_params/" + file_name;

    std::ifstream file(data_path, std::ios::in | std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    long file_size = file.tellg();
    unsigned long num_elements = static_cast<unsigned long>(file_size) / static_cast<unsigned long>(sizeof(dataType));
    file.seekg(0, std::ios::beg);

    std::vector<dataType> contents(num_elements);
    file.read(reinterpret_cast<char*>(contents.data()), file_size);

    return contents;
}

void
ProjectorByBinNiftyPETHelper::
set_up()
{
    if (_span < 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                "sinogram span not set.");

    if (_att < 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                "emission or transmission mode (att) not set.");

    // Read binaries
    if (_fname_li2rng.size()     == 0 ||
            _fname_li2sn.size()  == 0 ||
            _fname_li2nos.size() == 0 ||
            _fname_s2c.size()    == 0 ||
            _fname_aw2ali.size() == 0 ||
            _fname_crs.size()    == 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                 "not all filenames have been set.");

    _li2rng = read_binary_file<float>(_fname_li2rng);
    _li2sn  = read_binary_file<short>(_fname_li2sn );
    _li2nos = read_binary_file<char> (_fname_li2nos);
    _s2c    = read_binary_file<short>(_fname_s2c   );
    _aw2ali = read_binary_file<int>  (_fname_aw2ali);
    _crs    = read_binary_file<float>(_fname_crs   );
    _already_set_up = true;

    // Set up cnst
    _cnt.SPN      = _span;
    _cnt.RNG_STRT = 0;
    _cnt.RNG_END  = NRINGS;
    _cnt.VERBOSE  = false;
    _cnt.DEVID    = _devid;
    _cnt.NSN11    = NSINOS11;
    _cnt.NSEG0    = SEG0;

    switch(_cnt.SPN){
      case 11:
        _nsinos = _cnt.NSN11; break;
      case 0:
        _nsinos = _cnt.NSEG0; break;
      default:
        throw std::runtime_error("Unsupported span");
    }

    // isub
    _isub = std::vector<int>(unsigned(AW));
    for (unsigned i = 0; i<unsigned(AW); i++) _isub[i] = int(i);
}

void
ProjectorByBinNiftyPETHelper::
check_set_up() const
{
    if (!_already_set_up)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::check_set_up() "
                                 "Make sure filenames have been set and set_up has been run.");
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_image()
{
    return std::vector<float>(SZ_IMZ*SZ_IMX*SZ_IMY,0);
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_sinogram_no_gaps() const
{
    check_set_up();
    return std::vector<float>(_isub.size() * static_cast<unsigned long>(_nsinos), 0);
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_sinogram_with_gaps() const
{
    return std::vector<float>(NSBINS*NSANGLES*unsigned(_nsinos), 0);
}

void get_stir_indices_and_dims(int stir_dim[3], Coordinate3D<int> &min_indices, Coordinate3D<int> &max_indices, const DiscretisedDensity<3,float >&stir)
{
    if (!stir.get_regular_range(min_indices, max_indices))
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_input - "
                                 "expected image to have regular range.");
    for (int i=0; i<3; ++i)
        stir_dim[i] = max_indices[i + 1] - min_indices[i + 1] + 1;
}

unsigned convert_niftypet_im_3d_to_1d_idx(const unsigned x, const unsigned y, const unsigned z)
{
    return y*SZ_IMX*SZ_IMZ + x*SZ_IMZ + z;
}

unsigned
ProjectorByBinNiftyPETHelper::
convert_niftypet_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const
{
    return ang*NSBINS*unsigned(_nsinos) + bins*unsigned(_nsinos) + sino;
}

void check_im_sizes(const int stir_dim[3], const int np_dim[3])
{
    for (int i=0; i<3; ++i)
        if (stir_dim[i] != np_dim[i])
            throw std::runtime_error((boost::format(
                                      "ProjectorByBinNiftyPETHelper::check_im_sizes() - "
                                      "STIR image (%1%, %2%, %3%) should be == (%4%,%5%,%6%).")
                                      % stir_dim[0] % stir_dim[1] % stir_dim[2]
                                      % np_dim[0]   % np_dim[1]   % np_dim[2]).str());
}

void check_voxel_spacing(const DiscretisedDensity<3, float> &stir)
{
    // Requires image to be a VoxelsOnCartesianGrid
    const VoxelsOnCartesianGrid<float> &stir_vocg =
            dynamic_cast<const VoxelsOnCartesianGrid<float>&>(stir);
    const BasicCoordinate<3,float> stir_spacing = stir_vocg.get_grid_spacing();

    // Get NiftyPET image spacing (need to *10 for mm)
    float np_spacing[3] = { 10.f*SZ_VOXZ, 10.f*SZ_VOXY, 10.f*SZ_VOXY };

    for (unsigned i=0; i<3; ++i)
        if (std::abs(stir_spacing[int(i)+1] - np_spacing[i]) > 1e-4f)
            throw std::runtime_error((boost::format(
                                      "ProjectorByBinNiftyPETHelper::check_voxel_spacing() - "
                                      "STIR image (%1%, %2%, %3%) should be == (%4%,%5%,%6%).")
                                      % stir_spacing[1] % stir_spacing[2] % stir_spacing[3]
                                      % np_spacing[0]   % np_spacing[1]   % np_spacing[2]).str());
}

void
ProjectorByBinNiftyPETHelper::
convert_image_stir_to_niftyPET(std::vector<float> &np_vec, const DiscretisedDensity<3, float> &stir)
{
    // Get the dimensions of the input image
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    int stir_dim[3];
    get_stir_indices_and_dims(stir_dim,min_indices,max_indices,stir);

    // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
    // which at the time of writing was (127,320,320).
    const int np_dim[3] = {SZ_IMZ,SZ_IMX,SZ_IMY};
    check_im_sizes(stir_dim,np_dim);
    check_voxel_spacing(stir);

    // Copy data from STIR to NiftyPET image
    unsigned np_z, np_y, np_x, np_1d;
    for (int z = min_indices[1]; z <= max_indices[1]; z++) {
        for (int y = min_indices[2]; y <= max_indices[2]; y++) {
            for (int x = min_indices[3]; x <= max_indices[3]; x++) {
                // Convert the stir 3d index to a NiftyPET 1d index
                np_z = unsigned(z - min_indices[1]);
                np_y = unsigned(y - min_indices[2]);
                np_x = unsigned(x - min_indices[3]);
                np_1d = convert_niftypet_im_3d_to_1d_idx(np_x,np_y,np_z);
                np_vec[np_1d] = stir[z][y][x];
            }
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_image_niftyPET_to_stir(DiscretisedDensity<3,float> &stir, const std::vector<float> &np_vec)
{
    // Get the dimensions of the input image
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    int stir_dim[3];
    get_stir_indices_and_dims(stir_dim,min_indices,max_indices,stir);

    // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
    // which at the time of writing was (127,320,320).
    const int np_dim[3] = {SZ_IMZ,SZ_IMX,SZ_IMY};
    check_im_sizes(stir_dim,np_dim);
    check_voxel_spacing(stir);

    // Copy data from NiftyPET to STIR image
    unsigned np_z, np_y, np_x, np_1d;
    for (int z = min_indices[1]; z <= max_indices[1]; z++) {
        for (int y = min_indices[2]; y <= max_indices[2]; y++) {
            for (int x = min_indices[3]; x <= max_indices[3]; x++) {
                // Convert the stir 3d index to a NiftyPET 1d index
                np_z = unsigned(z - min_indices[1]);
                np_y = unsigned(y - min_indices[2]);
                np_x = unsigned(x - min_indices[3]);
                np_1d = convert_niftypet_im_3d_to_1d_idx(np_x,np_y,np_z);
                stir[z][y][x] = np_vec[np_1d];
            }
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_proj_data_stir_to_niftyPET(std::vector<float> &np_vec, const ProjData& stir) const
{
    // Get dimensions of STIR sinogram
    unsigned num_sinograms = unsigned(stir.get_num_sinograms());
    unsigned num_views     = unsigned(stir.get_num_views());
    unsigned num_tang_poss = unsigned(stir.get_num_tangential_poss());
    unsigned num_proj_data_elems = num_sinograms * num_views * num_tang_poss;

    // Make sure they're the same size
    if (np_vec.size() != num_proj_data_elems)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::convert_proj_data_stir_to_niftyPET "
                                 "NiftyPET and STIR sinograms are different sizes.");

    unsigned np_1d;
    for (unsigned ang=0; ang<num_views; ++ang) {
        for (unsigned bin=0; bin<num_tang_poss; ++bin) {
            for (unsigned sino=0; sino<num_sinograms; ++sino) {
                np_1d = convert_niftypet_proj_3d_to_1d_idx(ang,bin,sino);
                np_vec[np_1d] = -1.f;
                throw std::runtime_error("Need to do this: dont know how to access stir sinogram data");
            }
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_proj_data_niftyPET_to_stir(ProjData &stir, const std::vector<float> &np_vec) const
{
    // Get dimensions of STIR sinogram
    unsigned num_sinograms = unsigned(stir.get_num_sinograms());
    unsigned num_views     = unsigned(stir.get_num_views());
    unsigned num_tang_poss = unsigned(stir.get_num_tangential_poss());
    unsigned num_proj_data_elems = num_sinograms * num_views * num_tang_poss;

    // Make sure they're the same size
    if (np_vec.size() != num_proj_data_elems)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::convert_proj_data_stir_to_niftyPET "
                                 "NiftyPET and STIR sinograms are different sizes.");

    unsigned np_1d;
    for (unsigned ang=0; ang<num_views; ++ang) {
        for (unsigned bin=0; bin<num_tang_poss; ++bin) {
            for (unsigned sino=0; sino<num_sinograms; ++sino) {
                np_1d = convert_niftypet_proj_3d_to_1d_idx(ang,bin,sino);
                throw std::runtime_error("Need to do this: dont know how to access stir sinogram data");
            }
        }
    }
}

END_NAMESPACE_STIR
