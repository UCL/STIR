//
//
/*!

  \file
  \ingroup projection
  \ingroup NiftyPET

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

#include "stir/recon_buildblock/NiftyPET_projector/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/NiftyPET_projector/NiftyPETHelper.h"
#include "stir/ProjDataInMemory.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_array_functions.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
ForwardProjectorByBinNiftyPET::registered_name =
  "NiftyPET";

ForwardProjectorByBinNiftyPET::ForwardProjectorByBinNiftyPET() :
    _cuda_device(0), _cuda_verbosity(true), _use_truncation(false)
{
    this->_already_set_up = false;
}

ForwardProjectorByBinNiftyPET::~ForwardProjectorByBinNiftyPET()
{
}

void
ForwardProjectorByBinNiftyPET::
initialise_keymap()
{
  parser.add_start_key("Forward Projector Using NiftyPET Parameters");
  parser.add_stop_key("End Forward Projector Using NiftyPET Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
ForwardProjectorByBinNiftyPET::
set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr)
{
    ForwardProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));

    // Get span
    shared_ptr<const ProjDataInfoCylindricalNoArcCorr>
            proj_data_info_cy_no_ar_cor_sptr(
                dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(
                    proj_data_info_sptr));
    if (is_null_ptr(proj_data_info_cy_no_ar_cor_sptr))
        error("ForwardProjectorByBinNiftyPET: Failed casting to ProjDataInfoCylindricalNoArcCorr");
    int span = proj_data_info_cy_no_ar_cor_sptr->get_max_ring_difference(0) -
            proj_data_info_cy_no_ar_cor_sptr->get_min_ring_difference(0) + 1;

    // Initialise projected_data_sptr from this->_proj_data_info_sptr
    _projected_data_sptr.reset(
                new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), proj_data_info_sptr));

    // Set up the niftyPET binary helper
    _helper.set_scanner_type(proj_data_info_sptr->get_scanner_ptr()->get_type());
    _helper.set_cuda_device_id ( _cuda_device );
    _helper.set_span           ( static_cast<char>(span) );
    _helper.set_att(0);
    _helper.set_verbose(_cuda_verbosity);
    _helper.set_up();
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

void
ForwardProjectorByBinNiftyPET::
set_input(const DiscretisedDensity<3,float> & density)
{
    ForwardProjectorByBin::set_input(density);

    // Before forward projection, we enforce a truncation outside of the FOV.
    // This is because the NiftyPET FOV is smaller than the STIR FOV and this
    // could cause some voxel values to spiral out of control.
    if (_use_truncation)
        truncate_rim(*_density_sptr,17);

    // --------------------------------------------------------------- //
    //   STIR -> NiftyPET image data conversion
    // --------------------------------------------------------------- //

    std::vector<float> np_vec = _helper.create_niftyPET_image();
    _helper.convert_image_stir_to_niftyPET(np_vec,*_density_sptr);

    // --------------------------------------------------------------- //
    //   Forward projection
    // --------------------------------------------------------------- //

    std::vector<float> sino_no_gaps  = _helper.create_niftyPET_sinogram_no_gaps();
    _helper.forward_project(sino_no_gaps, np_vec);

    // --------------------------------------------------------------- //
    //   Put gaps back into sinogram
    // --------------------------------------------------------------- //

    std::vector<float> sino_w_gaps = _helper.create_niftyPET_sinogram_with_gaps();
    _helper.put_gaps(sino_w_gaps, sino_no_gaps);

    // --------------------------------------------------------------- //
    //   NiftyPET -> STIR projection data conversion
    // --------------------------------------------------------------- //

    _helper.convert_proj_data_niftyPET_to_stir(*_projected_data_sptr,sino_w_gaps);
}

END_NAMESPACE_STIR
