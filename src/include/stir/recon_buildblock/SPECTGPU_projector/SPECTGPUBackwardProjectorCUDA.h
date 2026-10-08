#pragma once

#include "stir/cuda_utilities.h"
#include "stir/RelatedViewgrams.h"
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/SPECTGPU_projector/ForwardProjectorByBinSPECTGPU.h"

START_NAMESPACE_STIR
// BPBuffers.h

void accumulate_image_omp_contrib(float* dev_image,
                                  const float* thread_image,
                                  unsigned int block_x,
                                  unsigned int block_y,
                                  unsigned int block_z,
                                  unsigned int grid_x,
                                  unsigned int grid_y,
                                  unsigned int grid_z,
                                  unsigned int image_dim);

void initialise_im_buffers(AllocatedStack& stack, bool do_atten, bool do_density = true);

void run_backward_projection_cuda(AllocatedStack& stack,
                                  const RelatedViewgrams<float>& stir_sino,
                                  bool do_atten,
                                  float coll_sigma0_cm,
                                  float _num_sigmas,
                                  float coll_slope,
                                  int num_views,
                                  unsigned int block_x,
                                  unsigned int block_y,
                                  unsigned int block_z,
                                  unsigned int grid_x,
                                  unsigned int grid_y,
                                  unsigned int grid_z,
                                  float spacing_x,
                                  float spacing_y,
                                  float spacing_z,
                                  float origin_x,
                                  float origin_y,
                                  float origin_z,
                                  int dim_x,
                                  int dim_y,
                                  int dim_z,
                                  int min_z,
                                  int min_y,
                                  int min_x);

END_NAMESPACE_STIR
