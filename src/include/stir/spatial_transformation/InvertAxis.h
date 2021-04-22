//
/*
 Copyright (C) 2019 National Physical Laboratory
 This file is part of STIR.
 
 SPDX-License-Identifier: Apache-2.0
 
 See STIR/LICENSE.txt for details
 */  
/*!
 \file
 \ingroup spatial_transformation
 
 \brief This invert the selected image axis.
 \author Daniel Deidda
*/

#ifndef __stir_InvertAxis_H__
#define __stir_InvertAxis_H__

#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR

/*! 
  \ingroup spatial_transformation
 
  \brief a utility class to "invert" an axis
  \warning this will reorder the voxel values without adjusting the geometric information.
*/
class InvertAxis{

public:

  //! transform the image
  /*! \a axis_name has to be x, y, z. Otherwise error() will be called.
   */
void
invert_axis(DiscretisedDensity<3, float> &inverted_image,
           const DiscretisedDensity<3, float> &input_image,
           const std::string & axis_name);

int
invert_axis_index(const int input_index,
                  const int size,
                  const std::string & axis_name);
};
END_NAMESPACE_STIR

#endif
