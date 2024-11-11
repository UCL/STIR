//
//
/*!
  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::GammaAutImageFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/GammaAutImageFilter2D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensity.h"
#include <algorithm>
#include <cmath>

START_NAMESPACE_STIR

template <typename elemT>
const char* const GammaAutImageFilter2D<elemT>::registered_name = "GammaAut2D";

template <typename elemT>
GammaAutImageFilter2D<elemT>::GammaAutImageFilter2D()
{
    set_defaults();
}

template <typename elemT>
void GammaAutImageFilter2D<elemT>::set_defaults()
{
    base_type::set_defaults();
}

template <typename elemT>
Succeeded GammaAutImageFilter2D<elemT>::virtual_set_up(const DiscretisedDensity<3, elemT>&)
{
    return Succeeded::yes;
}

template <typename elemT>
void GammaAutImageFilter2D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
    // Get the dimensions of the image
    const int sx = density.get_x_size();
    const int sy = density.get_y_size();
    const int sa = density.get_z_size();

    apply_gamma(dynamic_cast<VoxelsOnCartesianGrid<elemT>&>(density), sx, sy, sa);
}

template <typename elemT>
void GammaAutImageFilter2D<elemT>::apply_gamma(VoxelsOnCartesianGrid<elemT>& image, int sx, int sy, int sa) const
{
    const int min_x = image.get_min_x();
    const int min_y = image.get_min_y();
    float targetAverage = 0.25;

    for (int ia = 0; ia < sa; ia++)
    {
        float min_val = INFINITY, max_val = -INFINITY;

        // Step 1: Normalize the image slice
        for (int i = 0; i < sx; i++)
        {
            for (int j = 0; j < sy; j++)
            {
                min_val = std::min(image[ia][min_x + i][min_y + j], min_val);
                max_val = std::max(image[ia][min_x + i][min_y + j], max_val);
            }
        }

        for (int i = 0; i < sx; i++)
        {
            for (int j = 0; j < sy; j++)
            {
                image[ia][min_x + i][min_y + j] = (image[ia][min_x + i][min_y + j] - min_val) / (max_val - min_val);
            }
        }

        // Step 2: Compute the average pixel value for non-zero pixels
        int count = 0;
        float averagePixelValue = 0.0f;
        for (int i = 0; i < sx; i++)
        {
            for (int j = 0; j < sy; j++)
            {
                if (std::abs(image[ia][min_x + i][min_y + j]) > 0.1f)
                {
                    count++;
                    averagePixelValue += image[ia][min_x + i][min_y + j];
                }
            }
        }
        averagePixelValue /= count;

        // Step 3: Compute gamma value
        float gamma_val = 1.0f;
        if (averagePixelValue > 0.0f)
        {
            gamma_val = std::log(targetAverage) / std::log(averagePixelValue);
        }

        // Step 4: Apply gamma 
        for (int i = 0; i < sx; i++)
        {
            for (int j = 0; j < sy; j++)
            {
                if (std::abs(image[ia][min_x + i][min_y + j]) > 1e-6f)
                {
                    image[ia][min_x + i][min_y + j] = std::abs(image[ia][min_x + i][min_y + j]) > 1e-6
                                                ? std::pow(image[ia][min_x + i][min_y + j], gamma_val)
                                                : image[ia][min_x + i][min_y + j];
                }
            }
        }
 

        // Step 5: Denormalize the image slice
        for (int i = 0; i < sx; i++)
        {
            for (int j = 0; j < sy; j++)
            {
                image[ia][min_x + i][min_y + j] = image[ia][min_x + i][min_y + j] * (max_val - min_val) + min_val;
            }
        }
    }
}

template class GammaAutImageFilter2D<float>;

END_NAMESPACE_STIR

