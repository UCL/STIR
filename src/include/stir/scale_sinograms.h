//
//
/*
    Copyright (C) 2004 - 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief declaration of stir::scale_sinograms and stir::get_scale_factors_per_sinogram
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

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

//! find scale factors between two different sinograms
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

  Currently this function sets the scale factor or a sinogram to 1 (and calls warning())
  when the denominator gets too small.
*/
Array<2,float>
  get_scale_factors_per_sinogram(const ProjData& numerator_proj_data,
				 const ProjData& denominator_proj_data,
				 const ProjData& weights_proj_data);

END_NAMESPACE_STIR
