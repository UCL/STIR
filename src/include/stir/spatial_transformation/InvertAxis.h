//
/*
 Copyright (C) 2019 National Physical Laboratory
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
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

class InvertAxis{

public:

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
