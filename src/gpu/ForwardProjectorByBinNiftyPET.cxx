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
    ForwardProjectorByBin::set_up(proj_data_info_sptr, density_info_sptr);
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
        const int min_axial_pos_num, const int max_axial_pos_num,
        const int min_tangential_pos_num, const int max_tangential_pos_num)
{
    throw std::runtime_error("Need to use set_input() if wanting to use ForwardProjectorByBinNiftyPET.");
}

void
ForwardProjectorByBinNiftyPET::
actual_forward_project(RelatedViewgrams<float>& viewgrams,
        const int min_axial_pos_num, const int max_axial_pos_num,
        const int min_tangential_pos_num, const int max_tangential_pos_num)
{
//    if (min_axial_pos_num != _proj_data_info_sptr->get_min_axial_pos_num() ||
//         … )
//       error();

    viewgrams = _projected_data_sptr->get_related_viewgrams(
        viewgrams.get_basic_view_segment_num(), _symmetries_sptr);
}

void
ForwardProjectorByBinNiftyPET::
set_input(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr)
{
    _density_sptr.reset(density_sptr->clone());

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

    // Get dimensions of image
    int dim[3];
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    if (!_density_sptr->get_regular_range(min_indices, max_indices))
        throw std::runtime_error("ForwardProjectorByBinNiftyPET::set_input - "
                                 "expected image to have regular range.");
    for (int i=0; i<3; ++i)
        dim[i] = max_indices[i + 1] - min_indices[i + 1] + 1;
    // Create array for image and fill it - think carefully about index order!
    float *im_ptr = new float[dim[0]*dim[1]*dim[2]];
    for (int z = min_indices[1], i = 0; z <= max_indices[1]; z++)
		for (int y = min_indices[2]; y <= max_indices[2]; y++)
			for (int x = min_indices[3]; x <= max_indices[3]; x++, i++)
				im_ptr[i] = (*_density_sptr)[z][y][x];

    // Probably not necessary - delete after development if not used
    int num_segments  = this->_projected_data_sptr->get_num_segments();

    float * li2rng;
    short * li2sn;
    char * li2nos;
    short *s2c;
    int *aw2ali;
    float *crss;
    int *subs;
    int Nprj;
    int Naw;
    int n0crs;
    int n1crs;
    Cnst Cnt;
    char att;

    gpu_fprj(proj_data_ptr,im_ptr,li2rng,
        li2sn,li2nos,s2c,aw2ali,crss,
        subs,Nprj,Naw,n0crs, n1crs, Cnt, att);

    // Once finished, copy back
    _projected_data_sptr->fill_from(proj_data_ptr);

    // Delete created arrays
    delete [] proj_data_ptr;
    delete [] im_ptr;
}

END_NAMESPACE_STIR
