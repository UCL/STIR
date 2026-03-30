//
//
/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief non-inline implementations for stir::ForwardProjectorByBinSPECTGPU

  \author Daniel Deidda


*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SPECTGPU_projector/ForwardProjectorByBinSPECTGPU.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUHelper.h"
#include "stir/ProjDataInMemory.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_array_functions.h"
#include "stir/error.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char* const ForwardProjectorByBinSPECTGPU::registered_name = "SPECTGPU";

ForwardProjectorByBinSPECTGPU::ForwardProjectorByBinSPECTGPU()
    : _cuda_device(0),
      _cuda_verbosity(true),
      _use_truncation(false)
{
  this->_already_set_up = false;
}

ForwardProjectorByBinSPECTGPU::~ForwardProjectorByBinSPECTGPU()
{}

void
ForwardProjectorByBinSPECTGPU::initialise_keymap()
{
  parser.add_start_key("Forward Projector Using SPECTGPU Parameters");
  parser.add_stop_key("End Forward Projector Using SPECTGPU Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
ForwardProjectorByBinSPECTGPU::set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                      const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr)
{
  ForwardProjectorByBin::set_up(proj_data_info_sptr, density_info_sptr);
  check(*proj_data_info_sptr, *_density_sptr);

  auto& target_cast = dynamic_cast<const VoxelsOnCartesianGrid<elemT>&>(*target_sptr);
  auto sizes = target_cast.get_lengths();

  this->z_dim = sizes[1];
  this->y_dim = sizes[2];
  this->x_dim = sizes[3];

  // Set the thread block and grid dimensions using std::tuple
  this->block_dim.x = 8;
  this->block_dim.y = 8;
  this->block_dim.z = 8;

  this->grid_dim.x = (this->x_dim + this->block_dim.x - 1) / this->block_dim.x;
  this->grid_dim.y = (this->y_dim + this->block_dim.y - 1) / this->block_dim.y;
  this->grid_dim.z = (this->z_dim + this->block_dim.z - 1) / this->block_dim.z;

  //  Check if z_dim is 1 or only 2D is true and return an error if it is
  if (this->z_dim == 1 || this->only_2D)
    {
      error(" requires a 3D image and only works for a 3x3x3 neighbourhood");
      return Succeeded::no;
    }

//  {
//    if (this->d_kappa_data)
//      cudaFree(this->d_kappa_data);
//    auto kappa_ptr = this->get_kappa_sptr();
//    const bool do_kappa = !is_null_ptr(kappa_ptr);

  
  // Initialise projected_data_sptr from this->_proj_data_info_sptr
  _projected_data_sptr.reset(new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), proj_data_info_sptr));

  // Set up the SPECTGPU binary helper
  _helper.set_scanner_type(proj_data_info_sptr->get_scanner_ptr()->get_type());
  _helper.set_cuda_device_id(_cuda_device);
  _helper.set_att(0);
  _helper.set_verbose(_cuda_verbosity);
  _helper.set_up();
}


void
ForwardProjectorByBinSPECTGPU::actual_forward_project(
    RelatedViewgrams<float>& stir_sino,
        const DiscretisedDensity<3, float>& stir_image,
        const int min_ax,
        const int max_ax,
        const int min_tg,
        const int max_tg)
{
    //for all views in relateViewgram call the kernels
    dim3 cuda_block_dim(this->block_dim.x, this->block_dim.y, this->block_dim.z);
    dim3 cuda_grid_dim(this->grid_dim.x, this->grid_dim.y, this->grid_dim.z);
    viewgrams = _projected_data_sptr->get_related_viewgrams(viewgrams.get_basic_view_segment_num(), _symmetries_sptr);

    for (auto view=0; view<viewgram.get_num_viewgrams(),view++)
    {
        float* dev_image;
        cudaMalloc(&dev_image, stir_image.size_all() * sizeof(float));
        array_to_device(dev_image, stir_image);

        float* out_im;// need to copy on device?
        BasicCoordinate<3, int> min_ind, max_ind;

        stir_image.get_regular_range(min_ind, max_ind);

        const int min_z = min_ind[1];
        const int max_z = max_ind[1];

        const int min_y = min_ind[2];
        const int max_y = max_ind[2];

        const int min_x = min_ind[3];
        const int max_x = max_ind[3];

        int3 dim(max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1 );

        float angle_rad; //todo
        float3 spacing(,,);
        float3 origin(,,);

        rotateKernel_pull<<<cuda_grid_dim, cuda_block_dim>>>(dev_image,
                                       out_im,
                                       dim);

        forwardKernel<<<cuda_grid_dim, cuda_block_dim>>>(out_im,
                                     viewgrams[view],
                                     dim);

      }
      //  cudaMalloc(&this->cuda_image, stir_image_sptr->size_all() * sizeof(elemT));
    //  array_to_device(this->cuda_image, *stir_image_sptr);
    }


}

void
ForwardProjectorByBinSPECTGPU::actual_forward_project(
    RelatedViewgrams<float>& viewgrams, const int, const int, const int, const int)
{
      if (min_axial_pos_num != _proj_data_info_sptr->get_min_axial_pos_num() ||
  //         ... )
  //       error();
//for all views in relateViewgram call the kernels

  viewgrams = _projected_data_sptr->get_related_viewgrams(viewgrams.get_basic_view_segment_num(), _symmetries_sptr);
  for (auto view=0; view<viewgram.get_num_viewgrams(),view++)
  {
      cudaMalloc(&this->cuda_image, stir_image_sptr->size_all() * sizeof(elemT));
  }
  //  cudaMalloc(&this->cuda_image, stir_image_sptr->size_all() * sizeof(elemT));
//  array_to_device(this->cuda_image, *stir_image_sptr);
}

void
ForwardProjectorByBinSPECTGPU::set_input(const DiscretisedDensity<3, float>& density)
{
  ForwardProjectorByBin::set_input(density);

  // Before forward projection, we enforce a truncation outside of the FOV.
  // This is because the SPECTGPU FOV is smaller than the STIR FOV and this
  // could cause some voxel values to spiral out of control.
  if (_use_truncation)
    truncate_rim(*_density_sptr, 17);

  // --------------------------------------------------------------- //
  //   STIR -> SPECTGPU image data conversion
  // --------------------------------------------------------------- //

  std::vector<float> np_vec = _helper.create_SPECTGPU_image();
  _helper.convert_image_stir_to_SPECTGPU(np_vec, *_density_sptr);

  // --------------------------------------------------------------- //
  //   Forward projection
  // --------------------------------------------------------------- //

  std::vector<float> sino = _helper.create_SPECTGPU_sinogram();
  _helper.forward_project(sino, np_vec);

  std::vector<float> sino = _helper.create_SPECTGPU_sinogram();
  

  // --------------------------------------------------------------- //
  //   SPECTGPU -> STIR projection data conversion
  // --------------------------------------------------------------- //

  _helper.convert_proj_data_SPECTGPU_to_stir(*_projected_data_sptr, sino);
}

END_NAMESPACE_STIR
