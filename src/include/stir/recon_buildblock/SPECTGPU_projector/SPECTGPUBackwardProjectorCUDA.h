#pragma once

#include "stir/cuda_utilities.h"
#include "stir/RelatedViewgrams.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR
// BPBuffers.h
void allocate_im_buffers(float*& dev_image,
                         float*& dev_umap,
                         const DiscretisedDensity<3, float>& image,
                         const DiscretisedDensity<3, float>& umap,
                         bool do_atten);

void free_im_buffers(float* dev_image, float* dev_umap, bool do_atten);

void copy_im_to_stir(DiscretisedDensity<3, float>& image, const float* dev_image);

void copy_stir_im_to_dev(float* dev_image, const DiscretisedDensity<3, float>& image);

void run_backward_projection_cuda(float* dev_image,
                                  const RelatedViewgrams<float>& stir_sino,
                                  const float* dev_umap,
                                  bool do_atten,
                                  float coll_sigma0_cm,
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
