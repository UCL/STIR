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
  \ingroup scatter
  \brief A collection of functions to measure the scatter component  
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Pablo Aguiar

  $Date$
  $Revision$
*/

#include "stir/ProjData.h"
#include "stir/Array.h"

START_NAMESPACE_STIR


// give mask_radius_in_mm negative to ignore it
Array<2,float>
  scale_factors_per_sinogram(const ProjData& no_scatter_proj_data, 
			     const ProjData& scatter_proj_data, 
			     const ProjData& att_proj_data, 
			     const float attenuation_threshold,
#ifdef SCFOLD
			     const float mask_radius_in_mm
#else
			     const std::size_t back_off
#endif
			     );
Array<2,float>
  scale_factors_per_viewgram(const ProjData& no_scatter_proj_data, 
			     const ProjData& scatter_proj_data, 
			     const ProjData& att_proj_data, 
			     const float attenuation_threshold,
			     const float mask_radius_in_mm);

	void scale_scatter_per_sinogram(
		ProjData& scaled_scatter_proj_data, 		
		const ProjData& scatter_proj_data, 
		const Array<2,float> scale_factor_per_sinogram);


	void scale_scatter_per_viewgram(
		ProjData& scaled_scatter_proj_data, 		
		const ProjData& scatter_proj_data, 
		const Array<2,float> scale_factor_per_viewgram);

	/*void substract_scatter(
		ProjData& corrected_scatter_proj_data, 
		const shared_ptr<ProjData> & no_scatter_proj_data_sptr, 
		const shared_ptr<ProjData> & scatter_proj_data_sptr, 
		const ProjData& att_proj_data, 
		const float global_scatter_factor) ;*/


END_NAMESPACE_STIR
