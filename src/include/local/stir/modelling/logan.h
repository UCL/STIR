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

  \brief Declaration of logan functions

  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/
#include "stir/DiscretisedDensity.h"
#include "local/stir/DynamicDiscretisedDensity.h"
#include "local/stir/modelling/ParametricDiscretisedDensity.h"
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

/*! Applies the Ordinary-Least-Squares Fit, Not-Weighted, the same as Patlak implementation in STIR.*/
void apply_OLS_logan_to_images(ParametricVoxelsOnCartesianGrid & par_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    PlasmaData & plasma_data,
			    const float time_shift, 
			    const unsigned int starting_frame, 
			    const float bv, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected);

/*! Applies the Non-Linear-Least-Squares Fit, as proposed by Ogden 2003, Statistics in Medicine.*/
void apply_NLLS_logan_to_images(ParametricVoxelsOnCartesianGrid & par_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    PlasmaData & plasma_data,
			    const float time_shift, 
			    const unsigned int starting_frame, 
			    const float bv, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected);

END_NAMESPACE_STIR

