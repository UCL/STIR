//
//
/*!
  \file
  \ingroup projection
  \ingroup Parallelproj

  \brief non-inline implementations for stir::BackProjectorByBinParallelproj

  \author Richard Brown
  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2019, 2021 University College London
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

#include "stir/recon_buildblock/Parallelproj_projector/BackProjectorByBinParallelproj.h"
//#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_array_functions.h"
#include "stir/ProjDataInMemory.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
BackProjectorByBinParallelproj::registered_name =
  "Parallelproj";

BackProjectorByBinParallelproj::BackProjectorByBinParallelproj() :
    _cuda_device(0), _cuda_verbosity(true)
{
    this->_already_set_up = false;
}

BackProjectorByBinParallelproj::~BackProjectorByBinParallelproj()
{
}

void
BackProjectorByBinParallelproj::
initialise_keymap()
{
  parser.add_start_key("Back Projector Using Parallelproj Parameters");
  parser.add_stop_key("End Back Projector Using Parallelproj Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
BackProjectorByBinParallelproj::
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
        error("BackProjectorByBinParallelproj: Failed casting to ProjDataInfoCylindricalNoArcCorr");

    // Create sinogram
    _proj_data_to_backproject_sptr.reset(
                new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), proj_data_info_sptr));
}

const DataSymmetriesForViewSegmentNumbers *
BackProjectorByBinParallelproj::
get_symmetries_used() const
{
  if (!this->_already_set_up)
    error("BackProjectorByBin method called without calling set_up first.");
  return _symmetries_sptr.get();
}
#if 0
void
BackProjectorByBinParallelproj::
back_project(const ProjData& proj_data, int subset_num, int num_subsets)
{
    // Check the user has tried to project all data
    if (subset_num != 0 || num_subsets != 1)
        error("BackProjectorByBinParallelproj::back_project "
              "only works with all data (no subsets).");

    _helper.convert_proj_data_stir_to_niftyPET(_np_sino_w_gaps,proj_data);
}
#endif

void
BackProjectorByBinParallelproj::
get_output(DiscretisedDensity<3,float> &density) const
{

    std::vector<float> image_vec(density.size_all());
    // TODO 

    // --------------------------------------------------------------- //
    //   Parallelproj -> STIR image conversion
    // --------------------------------------------------------------- //
    std::copy(image_vec.begin(), image_vec.end(), density.begin_all());

}

void
BackProjectorByBinParallelproj::
start_accumulating_in_new_target()
{
    // Call base level
    BackProjectorByBin::start_accumulating_in_new_target();
    //  reset the Parallelproj sinogram
    _proj_data_to_backproject_sptr->fill(0.F);
}

void
BackProjectorByBinParallelproj::
actual_back_project(const RelatedViewgrams<float>& related_viewgrams,
                         const int, const int,
                         const int, const int)
{
  // TODO would have to check if only limited data is being backprojected
  _proj_data_to_backproject_sptr->set_related_viewgrams(related_viewgrams);
}

END_NAMESPACE_STIR
