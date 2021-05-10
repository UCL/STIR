//
//
/*!
  \file
  \ingroup projection
  \ingroup NiftyPET

  \brief non-inline implementations for stir::BackProjectorByBinNiftyPET

  \author Richard Brown
  
*/
/*
    Copyright (C) 2019, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/NiftyPET_projector/BackProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/NiftyPET_projector/NiftyPETHelper.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_array_functions.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
BackProjectorByBinNiftyPET::registered_name =
  "NiftyPET";

BackProjectorByBinNiftyPET::BackProjectorByBinNiftyPET() :
    _cuda_device(0), _cuda_verbosity(true), _use_truncation(false)
{
    this->_already_set_up = false;
}

BackProjectorByBinNiftyPET::~BackProjectorByBinNiftyPET()
{
}

void
BackProjectorByBinNiftyPET::
initialise_keymap()
{
  parser.add_start_key("Back Projector Using NiftyPET Parameters");
  parser.add_stop_key("End Back Projector Using NiftyPET Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
BackProjectorByBinNiftyPET::
set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr)
{
    BackProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));

    // Get span
    shared_ptr<const ProjDataInfoCylindricalNoArcCorr>
            proj_data_info_cy_no_ar_cor_sptr(
                dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(
                    proj_data_info_sptr));
    if (is_null_ptr(proj_data_info_cy_no_ar_cor_sptr))
        error("BackProjectorByBinNiftyPET: Failed casting to ProjDataInfoCylindricalNoArcCorr");
    int span = proj_data_info_cy_no_ar_cor_sptr->get_max_ring_difference(0) -
            proj_data_info_cy_no_ar_cor_sptr->get_min_ring_difference(0) + 1;

    // Set up the niftyPET binary helper
    _helper.set_cuda_device_id ( _cuda_device );
    _helper.set_scanner_type(proj_data_info_sptr->get_scanner_ptr()->get_type());
    _helper.set_span           ( static_cast<char>(span) );
    _helper.set_att(0);
    _helper.set_verbose(_cuda_verbosity);
    _helper.set_up();

    // Create sinogram
    _np_sino_w_gaps = _helper.create_niftyPET_sinogram_with_gaps();
}

const DataSymmetriesForViewSegmentNumbers *
BackProjectorByBinNiftyPET::
get_symmetries_used() const
{
  if (!this->_already_set_up)
    error("BackProjectorByBin method called without calling set_up first.");
  return _symmetries_sptr.get();
}

void
BackProjectorByBinNiftyPET::
back_project(const ProjData& proj_data, int subset_num, int num_subsets)
{
    // Check the user has tried to project all data
    if (subset_num != 0 || num_subsets != 1)
        error("BackProjectorByBinNiftyPET::back_project "
              "only works with all data (no subsets).");

    _helper.convert_proj_data_stir_to_niftyPET(_np_sino_w_gaps,proj_data);
}

void
BackProjectorByBinNiftyPET::
get_output(DiscretisedDensity<3,float> &density) const
{
    // --------------------------------------------------------------- //
    //   Remove gaps from sinogram
    // --------------------------------------------------------------- //

    std::vector<float> sino_no_gaps = _helper.create_niftyPET_sinogram_no_gaps();
    _helper.remove_gaps(sino_no_gaps, _np_sino_w_gaps);

    // --------------------------------------------------------------- //
    //   Back project
    // --------------------------------------------------------------- //

    std::vector<float> np_im = _helper.create_niftyPET_image();
    _helper.back_project(np_im,sino_no_gaps);

    // --------------------------------------------------------------- //
    //   NiftyPET -> STIR image conversion
    // --------------------------------------------------------------- //

    _helper.convert_image_niftyPET_to_stir(density,np_im);

    // After the back projection, we enforce a truncation outside of the FOV.
    // This is because the NiftyPET FOV is smaller than the STIR FOV and this
    // could cause some voxel values to spiral out of control.
    if (_use_truncation)
        truncate_rim(density,17);
}

void
BackProjectorByBinNiftyPET::
start_accumulating_in_new_target()
{
    // Call base level
    BackProjectorByBin::start_accumulating_in_new_target();
    // Also reset the NiftyPET sinogram
    _np_sino_w_gaps = _helper.create_niftyPET_sinogram_with_gaps();
}

void
BackProjectorByBinNiftyPET::
actual_back_project(const RelatedViewgrams<float>& related_viewgrams,
                         const int, const int,
                         const int, const int)
{
    for(stir::RelatedViewgrams<float>::const_iterator iter = related_viewgrams.begin(); iter != related_viewgrams.end(); ++iter)
        _helper.convert_viewgram_stir_to_niftyPET(_np_sino_w_gaps,*iter);
}

END_NAMESPACE_STIR
