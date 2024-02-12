//
//
/*!
  \file
  \ingroup Parallelproj

  \brief non-inline implementations for stir::BackProjectorByBinParallelproj

  \author Richard Brown
  \author Kris Thielemans
  \author Nicole Jurjew
  
*/
/*
    Copyright (C) 2019, 2021, 2024 University College London
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
#include "stir/TOF_conversions.h"
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
#include "stir/error.h"
#include "stir/stream.h"
#include <iostream>


START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
BackProjectorByBinParallelproj::registered_name =
  "Parallelproj";

BackProjectorByBinParallelproj::BackProjectorByBinParallelproj() :
    _cuda_verbosity(true), _num_gpu_chunks(1)
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
  parser.add_key("verbosity", &_cuda_verbosity);
  parser.add_key("num_gpu_chunks", &_num_gpu_chunks);
}

void
BackProjectorByBinParallelproj::
set_defaults()
{
  _cuda_verbosity = true;
  _num_gpu_chunks = 1;
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

static void TOF_transpose(std::vector<float>& mem_for_PP_back, const float *STIR_mem, const shared_ptr<const detail::ParallelprojHelper> _helper, const long long offset)
{
  const auto num_tof_bins = static_cast<unsigned>(_helper->num_tof_bins);
  for (unsigned tof_idx = 0; tof_idx < num_tof_bins; ++tof_idx)
    for (long long lor_idx = 0; lor_idx < _helper->num_lors; ++lor_idx)
      {
        mem_for_PP_back[lor_idx * num_tof_bins + tof_idx] = STIR_mem[offset + tof_idx * _helper->num_lors + lor_idx];
      }
}

void
BackProjectorByBinParallelproj::
get_output(DiscretisedDensity<3,float> &density) const
{

    std::vector<float> image_vec(density.size_all());
    // create an alias for the projection data
    const ProjDataInMemory& p(*_proj_data_to_backproject_sptr);

    info("Calling parallelproj backprojector", 2);

#ifdef parallelproj_built_with_CUDA

    long long num_lors_per_chunk_floor = _helper->num_lors / _num_gpu_chunks;
    long long remainder = _helper->num_lors % _num_gpu_chunks;

    long long num_lors_per_chunk;
    long long offset = 0;

    // send image to all visible CUDA devices
    float** image_on_cuda_devices = copy_float_array_to_all_devices(image_vec.data(), _helper->num_image_voxel);

    // do (chuck-wise) back projection on the CUDA devices 
    for(int chunk_num = 0; chunk_num < _num_gpu_chunks; chunk_num++){
      if(chunk_num < remainder){
        num_lors_per_chunk = num_lors_per_chunk_floor + 1;
      }
      else{
        num_lors_per_chunk = num_lors_per_chunk_floor;
      }
      if (p.get_proj_data_info_sptr()->is_tof_data())
        {
          info("running the CUDA version of parallelproj, about to call function joseph3d_back_tof_sino_cuda", 2);

          std::vector<float> mem_for_PP_back(num_lors_per_chunk * _helper->num_tof_bins);
          const float* STIR_mem = p.get_const_data_ptr();

          TOF_transpose(mem_for_PP_back, STIR_mem, _helper, offset);

          // info("created object mem_for_PP_img", 2);
          joseph3d_back_tof_sino_cuda(_helper->xend.data() + 3 * offset, _helper->xstart.data() + 3 * offset,
                                      image_on_cuda_devices, _helper->origin.data(), _helper->voxsize.data(),
                                      mem_for_PP_back.data(), // p.get_const_data_ptr() + offset* num_tof_bins,
                                      num_lors_per_chunk, _helper->imgdim.data(), _helper->tofbin_width, &_helper->sigma_tof,
                                      &_helper->tofcenter_offset,
                                      4, // float n_sigmas
                                      _helper->num_tof_bins,
                                      0, // unsigned char lor_dependent_sigma_tof
                                      0, // unsigned char lor_dependent_tofcenter_offset
                                      64 // threadsperblock
          );
          if (chunk_num != _num_gpu_chunks - 1)
            p.release_const_data_ptr();
        }
      else
      {
      joseph3d_back_cuda(_helper->xstart.data() + 3*offset,
                         _helper->xend.data() + 3*offset,
                         image_on_cuda_devices,
                         _helper->origin.data(),
                         _helper->voxsize.data(),
                         p.get_const_data_ptr() + offset,
                         num_lors_per_chunk,
                         _helper->imgdim.data(),
                         /*threadsperblock*/ 64
                         );
      }
      offset += num_lors_per_chunk;
    }

    // sum backprojected images on the first CUDA device
    sum_float_arrays_on_first_device(image_on_cuda_devices, _helper->num_image_voxel);

    // copy summed image back to host
    get_float_array_from_device(image_on_cuda_devices, _helper->num_image_voxel, 0, image_vec.data());

    // free image array from CUDA devices
    free_float_array_on_all_devices(image_on_cuda_devices);

#else
    if (this->_proj_data_info_sptr->is_tof_data() == 1)
      {
        std::vector<float> mem_for_PP_back(_helper->num_lors * _helper->num_tof_bins);
        const float* STIR_mem = p.get_const_data_ptr();

        TOF_transpose(mem_for_PP_back, STIR_mem, _helper, offset);

        joseph3d_back_tof_sino(_helper->xend.data(), _helper->xstart.data(), image_vec.data(), _helper->origin.data(),
                               _helper->voxsize.data(), mem_for_PP_back, num_lors, _helper->imgdim.data(), tofbin_width,
                               &sigma_tof, &tofcenter_offset,
                               4, // float n_sigmas,
                               _projected_data_sptr->get_proj_data_info_sptr()->get_num_tof_poss(),
                               0, //  unsigned char lor_dependent_sigma_tof
                               0  // unsigned char lor_dependent_tofcenter_offset
        );
      }
    else{
    joseph3d_back(_helper->xstart.data(),
                  _helper->xend.data(),
                  image_vec.data(),
                  _helper->origin.data(),
                  _helper->voxsize.data(),
                  p.get_const_data_ptr(),
                  static_cast<long long>(p.get_proj_data_info_sptr()->size_all()),
                  _helper->imgdim.data());
    }
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
