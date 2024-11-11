//
//
/*!
  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::MedianAutImageFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/MedianAutImageFilter2D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensity.h"
#include <algorithm>
#include <vector>

START_NAMESPACE_STIR

template <typename elemT>
const char* const MedianAutImageFilter2D<elemT>::registered_name = "MedianAut2D";

template <typename elemT>
MedianAutImageFilter2D<elemT>::MedianAutImageFilter2D()
{
    set_defaults();
}

template <typename elemT>
void MedianAutImageFilter2D<elemT>::set_defaults()
{
    base_type::set_defaults();
}

template <typename elemT>
Succeeded MedianAutImageFilter2D<elemT>::virtual_set_up(const DiscretisedDensity<3, elemT>&)
{
    return Succeeded::yes;
}

template <typename elemT>
void MedianAutImageFilter2D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
    // Get the dimensions of the image
    const int sx = density.get_x_size();
    const int sy = density.get_y_size();
    const int sa = density.get_z_size();

    apply_median_filter(dynamic_cast<VoxelsOnCartesianGrid<elemT>&>(density), sx, sy, sa);
}

template <typename elemT>
void MedianAutImageFilter2D<elemT>::apply_median_filter(VoxelsOnCartesianGrid<elemT>& image, int sx, int sy, int sa) const
{
    const int min_x = image.get_min_x();
    const int min_y = image.get_min_y();
    const int filter_size = 3;
    const int offset = filter_size / 2;
    const int len = 4; // Median index for a 3x3 filter
    std::vector<double> neighbors(filter_size * filter_size, 0);

    for (int ia = 0; ia < sa; ia++)
    {
        for (int i = 0; i < sx; i++)
        {
            for (int j = 0; j < sy; j++)
            {
                if (i == 0 || i == sx - 1 || j == 0 || j == sy - 1)
                    continue;

                // Collect neighbors for median computation
                for (int k = -offset; k <= offset; k++)
                {
                    for (int l = -offset; l <= offset; l++)
                    {
                        neighbors[(k + offset) * filter_size + l + offset] =
                            image[ia][min_x + i + k][min_y + j + l];
                    }
                }

                // Sort neighbors and find the median value
                std::sort(neighbors.begin(), neighbors.end());
                image[ia][min_x + i][min_y + j] = neighbors[len];
            }
        }
    }
    cerr << "Median filter complete" << endl;
}

template class MedianAutImageFilter2D<float>;

END_NAMESPACE_STIR

