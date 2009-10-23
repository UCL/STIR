//
// $Id$
//
/*
    Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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
  \ingroup projdata
  \brief declaration of stir::scale_sinograms and stir::get_scale_factors_per_sinogram
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/Array.h"

START_NAMESPACE_STIR
class ProjData;
class Succeeded;

//! apply a scale factor for every sinogram
/*! \ingroup projdata
  \param[out] output_proj_data is were the new sinograms will be stored
  \param[in] input_proj_data input
  \param[in] scale_factors_per_sinogram array with the scale factors. The first index
       corresponds to segments, the second to axial positions.
  \return indicates if writing failed or not
*/
Succeeded scale_sinograms(ProjData& output_proj_data, 		
			  const ProjData& input_proj_data, 
			  const Array<2,float> scale_factors_per_sinogram);

/*! find scale factors between two different sinograms
/*! \ingroup projdata
  \param[in] numerator_proj_data input data
  \param[in] denominator_proj_data input data
  \param[in] weights_proj_data weights to be taken into account
  \return scale_factors_per_sinogram array with the scale factors. The first index
       corresponds to segments, the second to axial positions.

  scale factors are found for every sinogram such that
  \code
    scale_factor = sum(numerator * weights) / sum(denominator * weights)
  \endcode

  \todo currently this function calls error() when the denominator gets too small
*/
Array<2,float>
  get_scale_factors_per_sinogram(const ProjData& numerator_proj_data,
				 const ProjData& denominator_proj_data,
				 const ProjData& weights_proj_data);

END_NAMESPACE_STIR
