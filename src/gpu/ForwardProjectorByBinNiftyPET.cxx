//
//
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ForwardProjectorByBinNiftyPET

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
#include "stir/gpu/ForwardProjectorByBinNiftyPET.h"
#include "stir/RelatedViewgrams.h"
#include <prjf.h>

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
ForwardProjectorByBinNiftyPET::registered_name =
  "NiftyPET";

ForwardProjectorByBinNiftyPET::ForwardProjectorByBinNiftyPET()
{
    this->_already_set_up = false;
}

ForwardProjectorByBinNiftyPET::~ForwardProjectorByBinNiftyPET()
{
}

void
ForwardProjectorByBinNiftyPET::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr)
{
    ForwardProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*this->_proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_sptr, density_info_sptr));

    // Initialise projected_data_sptr from this->_proj_data_info_sptr
    _projected_data_sptr.reset(
                new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), this->_proj_data_info_sptr));
}

const DataSymmetriesForViewSegmentNumbers *
ForwardProjectorByBinNiftyPET::
get_symmetries_used() const
{
  if (!this->_already_set_up)
    error("ForwardProjectorByBin method called without calling set_up first.");
  return _symmetries_sptr.get();
}

void
ForwardProjectorByBinNiftyPET::
actual_forward_project(RelatedViewgrams<float>&,
      const DiscretisedDensity<3,float>&,
        const int, const int,
        const int, const int)
{
    throw std::runtime_error("Need to use set_input() if wanting to use ForwardProjectorByBinNiftyPET.");
}

void
ForwardProjectorByBinNiftyPET::
actual_forward_project(RelatedViewgrams<float>& viewgrams,
        const int, const int,
        const int, const int)
{
//    if (min_axial_pos_num != _proj_data_info_sptr->get_min_axial_pos_num() ||
//         … )
//       error();

    viewgrams = _projected_data_sptr->get_related_viewgrams(
        viewgrams.get_basic_view_segment_num(), _symmetries_sptr);
}

template <class dataType>
static std::vector<dataType>
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
ForwardProjectorByBinNiftyPET::
set_input(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr)
{

    // TODO: required?
    _density_sptr.reset(density_sptr->clone());

    // --------------------------------------------------------------- //
    //   Set up the image
    // --------------------------------------------------------------- //

    // Get the dimensions of the input image
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    if (!_density_sptr->get_regular_range(min_indices, max_indices))
        throw std::runtime_error("ForwardProjectorByBinNiftyPET::set_input - "
                                 "expected image to have regular range.");
    int stir_dim[3];
    for (int i = 0; i < 3; i++)
        stir_dim[i] = max_indices[i + 1] - min_indices[i + 1] + 1;

    // NiftyPET requires the image to be (z,x,y)=(128,320,320)
    const int np_dim[3] = {128,320,320};

    // Check that stir image is <= niftyPET image in all dimensions
    for (int i=0; i<3; ++i)
        if (stir_dim[i] > np_dim[i])
            throw std::runtime_error("STIR image should be smaller than (128,320,320).");

#if 0
    // COULD USE THE ZOOM FUNCTIONALITY FOR STIR DIMS > NP DIMS
    // image needs to be VoxelsOnCartesianGrid
    shared_ptr<VoxelsOnCartesianGrid<float> > orig_image =
            dynamic_pointer_cast<VoxelsOnCartesianGrid<float> >(density_sptr);
    const CartesianCoordinate3D<float> zoom(1.f,1.f,1.f);
    const CartesianCoordinate3D<int>   new_sizes(128, 320, 320);
    const CartesianCoordinate3D<float> offsets_in_mm(0.f,0.f,0.f);
    const ZoomOptions zoom_options = ZoomOptions::preserve_sum;
    const VoxelsOnCartesianGrid<float> &resized_image =
      zoom_image(*orig_image, zoom, offsets_in_mm, new_sizes, zoom_options);
#endif

    // Create the NityPET image and fill with zeroes
    float *im_ptr = new float[np_dim[0] * np_dim[1] * np_dim[2]];
    std::fill( im_ptr, im_ptr + sizeof( im_ptr ), 0.f );

    // Copy data from STIR to NiftyPET image
    int np_z, np_y, np_x, np_1d;
    for (int z = min_indices[1]; z <= max_indices[1]; z++) {
        for (int y = min_indices[2]; y <= max_indices[2]; y++) {
            for (int x = min_indices[3]; x <= max_indices[3]; x++) {
                // Convert the stir 3d index to a NiftyPET 1d index
                np_z = z - min_indices[1];
                np_y = y - min_indices[2];
                np_x = x - min_indices[3];
                np_1d = np_z*np_dim[0]*np_dim[1] + np_y * np_dim[1] + np_x;
                im_ptr[np_1d] = (*_density_sptr)[z][y][x];
            }
        }
    }

    // --------------------------------------------------------------- //
    //   Other arguments
    // --------------------------------------------------------------- //

    Cnst Cnt;
    Cnt.SPN = static_cast<char>(11);
    Cnt.RNG_STRT = static_cast<char>(0);
    Cnt.RNG_END = static_cast<char>(64);
    Cnt.VERBOSE = false;
    Cnt.DEVID = static_cast<char>(0);
    Cnt.NSN11 = 837;
    Cnt.NSEG0 = 127;

    int nsinos;
    switch(Cnt.SPN){
      case 11:
        nsinos = Cnt.NSN11; break;
      case 0:
        nsinos = Cnt.NSEG0; break;
      default:
        throw std::runtime_error("Unsupported span");
    }

    int Naw = 68516;  // len(txLUT["aw2ali"]) - number of active bins in 2D sino
    int n0crs = 4;    // txLUT["crs"].shape[0]
    int n1crs = 504;  // txLUT["crs"].shape[1]
    char att = 0;     // whether to exp{-result} for attenuation maps

    std::vector<int> isub;  // TODO: expose as argument?
    if (isub.size() == 0) {
      isub = std::vector<int>(unsigned(Naw));
      for (unsigned i = 0; i<unsigned(Naw); i++) isub[i] = int(i);
    }
    std::vector<float> sinog(isub.size() * static_cast<unsigned long>(nsinos), 0);

    // Read the binary files if not already done.
    if (aw2ali.size() == 0) {
        aw2ali = read_binary_file<int>  ("aw2ali.dat");
        li2rng = read_binary_file<float>("li2rng.dat");
        li2sn  = read_binary_file<short>("li2sn.dat" );
        li2nos = read_binary_file<char> ("li2nos.dat");
        s2c    = read_binary_file<short>("s2c.dat"   );
        crss   = read_binary_file<float>("crss.dat"  );
    }

    // --------------------------------------------------------------- //
    //   Do the forward projection!
    // --------------------------------------------------------------- //

    gpu_fprj(sinog.data(), im_ptr,
        li2rng.data(), li2sn.data(), li2nos.data(), s2c.data(), aw2ali.data(), crss.data(),
        isub.data(), int(isub.size()),
        Naw, n0crs, n1crs,
        Cnt, att);

    // --------------------------------------------------------------- //
    //   NiftyPET -> STIR projection data conversion
    // --------------------------------------------------------------- //

    // Once finished, copy back
    // TODO: convert NiftyPET sinog.data() => STIR proj_data_ptr

    // Get dimensions of sinogram
    int num_sinograms = _projected_data_sptr->get_num_sinograms();
    int num_views     = _projected_data_sptr->get_num_views();
    int num_tang_poss = _projected_data_sptr->get_num_tangential_poss();
    int num_proj_data_elems = num_sinograms * num_views * num_tang_poss;
    // Create array for sinogram and fill it
    float *proj_data_ptr = new float[num_proj_data_elems];
    // Necessary?
    for (int i=0; i<num_proj_data_elems; ++i)
        proj_data_ptr[i] = 0.F;
    _projected_data_sptr->fill_from(proj_data_ptr);

    // --------------------------------------------------------------- //
    //   Delete arrays
    // --------------------------------------------------------------- //

    // Delete created arrays
    delete [] proj_data_ptr;
    delete [] im_ptr;
}

END_NAMESPACE_STIR
