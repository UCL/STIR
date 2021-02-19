//
//
/*!

  \file
  \ingroup Parallelproj

  \brief non-inline implementations for stir::ForwardProjectorByBinParallelproj

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

#include "stir/recon_buildblock/Parallelproj_projector/ForwardProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"
#include "stir/ProjDataInMemory.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/info.h"
#ifdef parallelproj_built_with_CUDA
#include "parallelproj_cuda.h"
#else
#include "parallelproj_c.h"
#endif

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
ForwardProjectorByBinParallelproj::registered_name =
  "Parallelproj";

ForwardProjectorByBinParallelproj::ForwardProjectorByBinParallelproj() :
    _cuda_device(0), _cuda_verbosity(true), _use_truncation(false)
{
    this->_already_set_up = false;
    this->_do_not_setup_helper = false;
}

ForwardProjectorByBinParallelproj::~ForwardProjectorByBinParallelproj()
{
}

void
ForwardProjectorByBinParallelproj::
initialise_keymap()
{
  parser.add_start_key("Forward Projector Using Parallelproj Parameters");
  parser.add_stop_key("End Forward Projector Using Parallelproj Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
ForwardProjectorByBinParallelproj::set_helper(shared_ptr<detail::ParallelprojHelper> helper)
{
  this->_helper = helper;
  this->_do_not_setup_helper = true;
}

void
ForwardProjectorByBinParallelproj::
set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr)
{
    ForwardProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));

#if 0
    shared_ptr<const ProjDataInfoCylindricalNoArcCorr>
            proj_data_info_cy_no_ar_cor_sptr(
                dynamic_pointer_cast<const ProjDataInfoCylindricalNoArcCorr>(
                    proj_data_info_sptr));
    if (is_null_ptr(proj_data_info_cy_no_ar_cor_sptr))
        error("ForwardProjectorByBinParallelproj: Failed casting to ProjDataInfoCylindricalNoArcCorr");
#endif  
    // Initialise projected_data_sptr from this->_proj_data_info_sptr
    _projected_data_sptr.reset(
                new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), proj_data_info_sptr));
    if (!this->_do_not_setup_helper)
      _helper = std::make_shared<detail::ParallelprojHelper>(*proj_data_info_sptr, *density_info_sptr);

}

const DataSymmetriesForViewSegmentNumbers *
ForwardProjectorByBinParallelproj::
get_symmetries_used() const
{
  if (!this->_already_set_up)
    error("ForwardProjectorByBin method called without calling set_up first.");
  return _symmetries_sptr.get();
}

void
ForwardProjectorByBinParallelproj::
actual_forward_project(RelatedViewgrams<float>&,
      const DiscretisedDensity<3,float>&,
        const int, const int,
        const int, const int)
{
    throw std::runtime_error("Need to use set_input() if wanting to use ForwardProjectorByBinParallelproj.");
}

void
ForwardProjectorByBinParallelproj::
actual_forward_project(RelatedViewgrams<float>& viewgrams,
        const int, const int,
        const int, const int)
{
//    if (min_axial_pos_num != _proj_data_info_sptr->get_min_axial_pos_num() ||
//         ... )
//       error();

    viewgrams = _projected_data_sptr->get_related_viewgrams(
        viewgrams.get_basic_view_segment_num(), _symmetries_sptr);
}

void
ForwardProjectorByBinParallelproj::
set_input(const DiscretisedDensity<3,float> & density)
{
    ForwardProjectorByBin::set_input(density);


    std::vector<float> image_vec(density.size_all());
    std::copy(density.begin_all(), density.end_all(), image_vec.begin());

    // need to set output to zero as parallelproj accumulates
    _projected_data_sptr->fill(0.F);

    info("Calling parallelproj forward",2);
#ifdef parallelproj_built_with_CUDA
    joseph3d_fwd_cuda(_helper->xstart.data(),
                      _helper->xend.data(),
                      image_vec.data(),
                      _helper->origin.data(),
                      _helper->voxsize.data(),
                      _projected_data_sptr->get_data_ptr(),
                      static_cast<long long>(_projected_data_sptr->get_proj_data_info_sptr()->size_all()),
                      _helper->imgdim.data(),
                      /*threadsperblock*/ 64,
                      /*num_devices*/ -1);
);
#else
    joseph3d_fwd(_helper->xstart.data(),
                  _helper->xend.data(),
                  image_vec.data(),
                  _helper->origin.data(),
                  _helper->voxsize.data(),
                  _projected_data_sptr->get_data_ptr(),
                  static_cast<long long>(_projected_data_sptr->get_proj_data_info_sptr()->size_all()),
                  _helper->imgdim.data());
#endif
    info("done", 2);

    _projected_data_sptr->release_data_ptr();
}

END_NAMESPACE_STIR
