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

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/Parallelproj_projector/ForwardProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"
#include "stir/ProjDataInMemory.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/info.h"
#include "stir/recon_array_functions.h"
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
    _cuda_device(0), _cuda_verbosity(true), _use_truncation(true)
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
actual_forward_project(RelatedViewgrams<float>& viewgrams,
                       const int min_axial_pos_num, const int max_axial_pos_num,
                       const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  if ((min_axial_pos_num != this->_proj_data_info_sptr->get_min_axial_pos_num(viewgrams.get_basic_segment_num())) ||
      (max_axial_pos_num != this->_proj_data_info_sptr->get_max_axial_pos_num(viewgrams.get_basic_segment_num())) ||
      (min_tangential_pos_num != this->_proj_data_info_sptr->get_min_tangential_pos_num()) ||
      (max_tangential_pos_num != this->_proj_data_info_sptr->get_max_tangential_pos_num()))
    error("STIR wrapping of Parallelproj projectors current only handles projecting all data");

    viewgrams = _projected_data_sptr->get_related_viewgrams(
        viewgrams.get_basic_view_segment_num(), _symmetries_sptr);
}

void
ForwardProjectorByBinParallelproj::
set_input(const DiscretisedDensity<3,float> & density)
{
    ForwardProjectorByBin::set_input(density);

    // Before forward projection, we enforce a truncation outside of the FOV.
    // This is because the parallelproj projector seems to have some trouble at the edges and this
    // could cause some voxel values to spiral out of control.
    //if (_use_truncation)
      {
        const float radius = this->_proj_data_info_sptr->get_scanner_sptr()->get_inner_ring_radius();
        const float image_radius = _helper->voxsize[2]*_helper->imgdim[2]/2;
        truncate_rim(*_density_sptr, static_cast<int>(std::max((image_radius-radius) / _helper->voxsize[2],0.F)));
      }

    std::vector<float> image_vec(density.size_all());
    std::copy(_density_sptr->begin_all(), _density_sptr->end_all(), image_vec.begin());

#if 0
    // needed to set output to zero as parallelproj accumulates but is no longer the case
    _projected_data_sptr->fill(0.F);
#endif

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
