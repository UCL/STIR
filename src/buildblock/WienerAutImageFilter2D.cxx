#include "stir/WienerAutImageFilter2D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensity.h"
#include <vector>
#include <algorithm>

START_NAMESPACE_STIR

template <typename elemT>
const char* const WienerAutImageFilter2D<elemT>::registered_name = "WienerAut2D";

template <typename elemT>
WienerAutImageFilter2D<elemT>::WienerAutImageFilter2D()
{
    set_defaults();
}

template <typename elemT>
void WienerAutImageFilter2D<elemT>::set_defaults()
{
    base_type::set_defaults();
}

template <typename elemT>
Succeeded WienerAutImageFilter2D<elemT>::virtual_set_up(const DiscretisedDensity<3, elemT>&)
{
    return Succeeded::yes;
}

template <typename elemT>
void WienerAutImageFilter2D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
    // Get the dimensions of the image
    const int sx = density.get_x_size();
    const int sy = density.get_y_size();
    const int sa = density.get_z_size();

    apply_wiener_filter(dynamic_cast<VoxelsOnCartesianGrid<elemT>&>(density), sx, sy, sa);
}

template <typename elemT>
void WienerAutImageFilter2D<elemT>::apply_wiener_filter(VoxelsOnCartesianGrid<elemT>& image, int sx, int sy, int sa) const
{
    const int min_x = image.get_min_x();
    const int min_y = image.get_min_y();
    const int ws = 9;

    for (int ia = 0; ia < sa; ia++)
    {
        std::vector<std::vector<float>> localMean(sx, std::vector<float>(sy, 0.0f));
        std::vector<std::vector<float>> localVar(sx, std::vector<float>(sy, 0.0f));
        float noise = 0.0f;

        for (int i = 1; i < sx - 1; i++)
        {
            for (int j = 1; j < sy - 1; j++)
            {
                localMean[i][j] = 0;
                localVar[i][j] = 0;

                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        localMean[i][j] += image[ia][min_x + i + k][min_y + j + l];
                    }
                }
                localMean[i][j] /= ws;

                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        localVar[i][j] += image[ia][min_x + i + k][min_y + j + l] * image[ia][min_x + i + k][min_y + j + l];
                    }
                }
                localVar[i][j] = localVar[i][j] / ws - localMean[i][j] * localMean[i][j];
                noise += localVar[i][j];
            }
        }
        noise /= sx * sy;

        for (int i = 1; i < sx - 1; i++)
        {
            for (int j = 1; j < sy - 1; j++)
            {
                image[ia][min_x + i][min_y + j] =
                    (image[ia][min_x + i][min_y + j] - localMean[i][j]) / std::max(localVar[i][j], noise) *
                        std::max(localVar[i][j] - noise, 0.f) +
                    localMean[i][j];
            }
        }
    }
}

template class WienerAutImageFilter2D<float>;

END_NAMESPACE_STIR

