//
//
/*!
  \file
  \ingroup Parallelproj

  \brief non-inline implementations for stir::BackProjectorByBinParallelproj

  \author Richard Brown
  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2019, 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/Parallelproj_projector/BackProjectorByBinParallelproj.h"
#include "stir/recon_buildblock/Parallelproj_projector/ParallelprojHelper.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/ProjDataInfo.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_array_functions.h"
#include "stir/ProjDataInMemory.h"
#include "stir/LORCoordinates.h"
#include "stir/recon_array_functions.h"
#ifdef parallelproj_built_with_CUDA
#include "parallelproj_cuda.h"
#else
#include "parallelproj_c.h"
#endif
// for debugging, remove later
#include "stir/info.h"
#include "stir/stream.h"
#include <iostream>


START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
BackProjectorByBinParallelproj::registered_name =
  "Parallelproj";

BackProjectorByBinParallelproj::BackProjectorByBinParallelproj() :
    _cuda_device(0), _cuda_verbosity(true)
{
    this->_already_set_up = false;
    this->_do_not_setup_helper = false;
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
BackProjectorByBinParallelproj::set_helper(shared_ptr<detail::ParallelprojHelper> helper)
{
  this->_helper = helper;
  this->_do_not_setup_helper = true;
}

void
BackProjectorByBinParallelproj::
set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr)
{
    BackProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));

    // Create sinogram
    _proj_data_to_backproject_sptr.reset(
                new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), proj_data_info_sptr));

    if (!this->_do_not_setup_helper)
      _helper = std::make_shared<detail::ParallelprojHelper>(*proj_data_info_sptr, *density_info_sptr);
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
    // create an alias for the projection data
    const ProjDataInMemory& p(*_proj_data_to_backproject_sptr);

    info("Calling parallelproj backprojector", 2);

#ifdef parallelproj_built_with_CUDA
    joseph3d_back_cuda(_helper->xstart.data(),
                       _helper->xend.data(),
                       image_vec.data(),
                       _helper->origin.data(),
                       _helper->voxsize.data(),
                       p.get_const_data_ptr(),
                       static_cast<long long>(p.get_proj_data_info_sptr()->size_all()),
                       _helper->imgdim.data(),
                       /*threadsperblock*/ 64,
                       /*num_devices*/ -1);
#else
    joseph3d_back(_helper->xstart.data(),
                  _helper->xend.data(),
                  image_vec.data(),
                  _helper->origin.data(),
                  _helper->voxsize.data(),
                  p.get_const_data_ptr(),
                  static_cast<long long>(p.get_proj_data_info_sptr()->size_all()),
                  _helper->imgdim.data());
#endif
    info("done", 2);

    p.release_const_data_ptr();

    // --------------------------------------------------------------- //
    //   Parallelproj -> STIR image conversion
    // --------------------------------------------------------------- //
    std::copy(image_vec.begin(), image_vec.end(), density.begin_all());

    // After the back projection, we enforce a truncation outside of the FOV.
    // This is because the parallelproj projector seems to have some trouble at the edges and this
    // could cause some voxel values to spiral out of control.
    //if (_use_truncation)
      {
        const float radius = p.get_proj_data_info_sptr()->get_scanner_sptr()->get_inner_ring_radius();
        const float image_radius = _helper->voxsize[2]*_helper->imgdim[2]/2;
        truncate_rim(density, static_cast<int>(std::max((image_radius-radius) / _helper->voxsize[2],0.F)));
      }
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
                    const int min_axial_pos_num, const int max_axial_pos_num,
                    const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  if ((min_axial_pos_num != this->_proj_data_info_sptr->get_min_axial_pos_num(related_viewgrams.get_basic_segment_num())) ||
      (max_axial_pos_num != this->_proj_data_info_sptr->get_max_axial_pos_num(related_viewgrams.get_basic_segment_num())) ||
      (min_tangential_pos_num != this->_proj_data_info_sptr->get_min_tangential_pos_num()) ||
      (max_tangential_pos_num != this->_proj_data_info_sptr->get_max_tangential_pos_num()))
    error("STIR wrapping of Parallelproj projectors current only handles projecting all data");

  _proj_data_to_backproject_sptr->set_related_viewgrams(related_viewgrams);
}

END_NAMESPACE_STIR
