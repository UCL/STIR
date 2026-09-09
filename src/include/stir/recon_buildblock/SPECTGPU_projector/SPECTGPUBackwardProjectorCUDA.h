#pragma once

#include "stir/cuda_utilities.h"
#include "stir/RelatedViewgrams.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

void run_backward_projection_cuda(
    const RelatedViewgrams<float>& stir_sino,
    DiscretisedDensity<3,float>& stir_image,
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
