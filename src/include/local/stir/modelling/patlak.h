//
// $Id$
//
/*
    Copyright (C) 2005 - $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup modelling

  \brief Declaration of class stir::PlasmaData

  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/
#include "stir/DiscretisedDensity.h"
#include "local/stir/DynamicDiscretisedDensity.h"
#include "local/stir/modelling/PlasmaData.h"
#include "local/stir/modelling/PlasmaSample.h"
#include "local/stir/modelling/BloodFrame.h"
#include "local/stir/modelling/BloodFrameData.h"
#include "stir/common.h"
#include <vector>
#include <iostream> 
#include <cstring>
#include <iomanip> 
#include <fstream>

START_NAMESPACE_STIR

void apply_patlak_to_images_and_arterial_sampling(DiscretisedDensity<3,float>& y_intersection_image, 			     
			    DiscretisedDensity<3,float>& slope_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    PlasmaData & plasma_data,
			    const float time_shift, 
			    const int starting_frame, 
			    const float bv, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected);

void apply_patlak_to_images_plasma_based(DiscretisedDensity<3,float>& y_intersection_image, 			     
			    DiscretisedDensity<3,float>& slope_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    BloodFrameData & blood_frame_data,
			    const int starting_frame, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected);

END_NAMESPACE_STIR

