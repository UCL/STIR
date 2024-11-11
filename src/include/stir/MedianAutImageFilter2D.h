//
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Declaration of class stir::MedianAutImageFilter2D.h

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_MedianAutImageFilter2D_H__
#define __stir_MedianAutImageFilter2D_H__

#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup ImageProcessor
  \brief A class in the DataProcessor hierarchy that implements a 2D Median filter.
 */
template <typename elemT>
class MedianAutImageFilter2D : public RegisteredParsingObject<MedianAutImageFilter2D<elemT>,
                                                             DataProcessor<DiscretisedDensity<3, elemT>>,
                                                             DataProcessor<DiscretisedDensity<3, elemT>>>
{
private:
  typedef RegisteredParsingObject<MedianAutImageFilter2D<elemT>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>,
                                  DataProcessor<DiscretisedDensity<3, elemT>>>
      base_type;

public:
  static const char* const registered_name;

  //! Default constructor
  MedianAutImageFilter2D();

  //! Set default parameters
  void set_defaults() override;

  //! Applies the Median filter to the input density
  void virtual_apply(DiscretisedDensity<3, elemT>& density) const override;

protected:
  Succeeded virtual_set_up(const DiscretisedDensity<3, elemT>& density) override;

private:
  void apply_median_filter(VoxelsOnCartesianGrid<elemT>& image, int sx, int sy, int sa) const;
};

END_NAMESPACE_STIR

#endif // __stir_MedianAutImageFilter2D_H__

