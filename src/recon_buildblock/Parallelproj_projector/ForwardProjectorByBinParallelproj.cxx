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
#include "stir/error.h"
#include "stir/recon_array_functions.h"
#include "stir/utilities.h"
#include "stir/TOF_conversions.h"
#include <algorithm>
#ifdef parallelproj_built_with_CUDA
#include "parallelproj_cuda.h"
#else
#include "parallelproj_c.h"
#endif

START_NAMESPACE_STIR

template <class containerT>
void write_binary(const std::string& filename, const containerT& data)
{
  std::fstream s;
  open_write_binary(s, filename);
  const auto num_to_write =
    static_cast<std::streamsize>(data.size())* sizeof(typename containerT::value_type);
    s.write(reinterpret_cast<const char *>(data.data()), num_to_write);
  }

//////////////////////////////////////////////////////////
const char * const
ForwardProjectorByBinParallelproj::registered_name =
  "Parallelproj";

ForwardProjectorByBinParallelproj::ForwardProjectorByBinParallelproj() :
    _cuda_verbosity(true), _use_truncation(true), _num_gpu_chunks(1)
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
  parser.add_key("verbosity", &_cuda_verbosity);
  parser.add_key("num_gpu_chunks", &_num_gpu_chunks);
}

void
ForwardProjectorByBinParallelproj::
set_defaults()
{
  _cuda_verbosity = true;
  _use_truncation = true;
  _num_gpu_chunks = 1;
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
        viewgrams.get_basic_view_segment_num(), _symmetries_sptr, false, viewgrams.get_basic_timing_pos_num());
}

static void
TOF_transpose(float* STIR_mem, const std::vector<float>& mem_for_PP, const shared_ptr<const detail::ParallelprojHelper> _helper,
              const long long offset, const long long num_lors_per_chunk)
{
  const auto num_tof_bins = static_cast<unsigned>(_helper->num_tof_bins);
  for (unsigned tof_idx = 0; tof_idx < num_tof_bins; ++tof_idx)
    for (long long lor_idx = 0; lor_idx < num_lors_per_chunk; ++lor_idx)
      {
        STIR_mem[offset + tof_idx * _helper->num_lors + lor_idx] = mem_for_PP[lor_idx * num_tof_bins + tof_idx];
      }
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

    long long num_lors_per_chunk_floor = _helper->num_lors / _num_gpu_chunks; // num_lors=407, num_GPU_chunks = 10, --> num_lors_per_chunk_floor = 407/10 = 40
    long long remainder = _helper->num_lors % _num_gpu_chunks; // remainder = 7; so in 7 chunks I'll have 1 LOR more

    long long num_lors_per_chunk;
    long long offset = 0;


    // send image to all visible CUDA devices
    float** image_on_cuda_devices = copy_float_array_to_all_devices(image_vec.data(), _helper->num_image_voxel);

    // do (chuck-wise) projection on the CUDA devices 
    for(int chunk_num = 0; chunk_num < _num_gpu_chunks; chunk_num++){ // for chunk_num = 0 to 9, so this happens 10 times
      if(chunk_num < remainder){ // if chunk_num < 7 so from 0 to 6
        num_lors_per_chunk = num_lors_per_chunk_floor + 1; // num_lors_per_chunk = 40 + 1 = 41
      }
      else{ // if chunk_num >= 7 (so no remainder for chunks 7, 8, 9)
        num_lors_per_chunk = num_lors_per_chunk_floor; // num_lors_per_chunk = 40
      }

      if (_proj_data_info_sptr->is_tof_data() == 1)
        {

          std::vector<float> mem_for_PP(num_lors_per_chunk * _helper->num_tof_bins);
          joseph3d_fwd_tof_sino_cuda(_helper->xend.data() + 3 * offset, _helper->xstart.data() + 3 * offset,
                                     image_on_cuda_devices, _helper->origin.data(), _helper->voxsize.data(),
                                     mem_for_PP.data(),  // this is where the data is written to
                                     num_lors_per_chunk, // 41(40) , PP docu: "number of geometrical LORs";
                                     _helper->imgdim.data(), _helper->tofbin_width, &_helper->sigma_tof,
                                     &_helper->tofcenter_offset,
                                     4,                     // float n_sigmas
                                     _helper->num_tof_bins, // short n_tofbins
                                     0,                     // unsigned char lor_dependent_sigma_tof
                                     0,                     // unsigned char lor_dependent_tofcenter_offset
                                     64                     // threadsperblock
          );

          float* STIR_mem = _projected_data_sptr->get_data_ptr();

          TOF_transpose(STIR_mem, mem_for_PP, _helper, offset, num_lors_per_chunk);

          if (chunk_num != _num_gpu_chunks - 1)
            _projected_data_sptr->release_data_ptr();
          info("current proj max: "
               + std::to_string(*std::max_element(_projected_data_sptr->begin(), _projected_data_sptr->end())));
        }
      else
        {
          joseph3d_fwd_cuda(_helper->xstart.data() + 3 * offset, _helper->xend.data() + 3 * offset, image_on_cuda_devices,
                            _helper->origin.data(), _helper->voxsize.data(), _projected_data_sptr->get_data_ptr() + offset,
                            num_lors_per_chunk, _helper->imgdim.data(),
                            /*threadsperblock*/ 64);
          if (chunk_num != _num_gpu_chunks - 1)
            _projected_data_sptr->release_data_ptr();
        }
      offset += num_lors_per_chunk;
      }

    // free image array from CUDA devices
    free_float_array_on_all_devices(image_on_cuda_devices);

#else
    if (this->_proj_data_info_sptr->is_tof_data() == 1)
      {

        std::vector<float> mem_for_PP(_helper->num_lors * _helper->num_tof_bins);
        joseph3d_fwd_tof_sino(_helper->xend.data(), _helper->xstart.data(), image_vec.data(), _helper->origin.data(),
                              _helper->voxsize.data(), mem_for_PP.data(), _helper->num_lors, _helper->imgdim.data(),
                              _helper->tofbin_width, &_helper->sigma_tof, &_helper->tofcenter_offset,
                              4, // float n_sigmas,
                              _helper->num_tof_bins,
                              0, //  unsigned char lor_dependent_sigma_tof
                              0  // unsigned char lor_dependent_tofcenter_offset
        );

        float* STIR_mem = _projected_data_sptr->get_data_ptr();
        TOF_transpose(STIR_mem, mem_for_PP, _helper, offset, _helper->num_lors);
      }
    else
      {
        joseph3d_fwd(_helper->xstart.data(), _helper->xend.data(), image_vec.data(), _helper->origin.data(),
                     _helper->voxsize.data(), _projected_data_sptr->get_data_ptr(),
                     static_cast<long long>(_projected_data_sptr->get_proj_data_info_sptr()->size_all()), _helper->imgdim.data());
      }
#endif
    info("done", 2);

    _projected_data_sptr->release_data_ptr();
}

END_NAMESPACE_STIR
