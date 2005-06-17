//
// $Id$
//
/*
Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
//! Perform B-Splines Interpolation

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
*/
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
//#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"
#include "local/stir/interpolate_projdata.h"
#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"

using namespace BSpline;

START_NAMESPACE_STIR

/*Succeeded 
interpolate_projdata(ProjData& proj_data_out,
					 const ProjData& proj_data_in, const BSplineType this_type)
{
	BasicCoordinate<3, BSplineType> this_type_3; this_type_3[1]=this_type_3[2]=this_type_3[3]=this_type;
	interpolate_projdata(proj_data_out,proj_data_in,this_type_3);
	return Succeeded::yes;
}*/

Succeeded 
interpolate_projdata(ProjData& proj_data_out,
					 const ProjData& proj_data_in,
					 const BasicCoordinate<3, BSplineType> & this_type)
{
	const ProjDataInfo * const proj_data_in_info_ptr =
		dynamic_cast<ProjDataInfo const * >
		(proj_data_in.get_proj_data_info_ptr());
	const ProjDataInfo * const proj_data_out_info_ptr =
		dynamic_cast<ProjDataInfo const * >
		(proj_data_out.get_proj_data_info_ptr());
	
	const SegmentBySinogram<float> sino_3D_in = proj_data_in.get_segment_by_sinogram(0) ;
	
	BSplinesRegularGrid<3, float, float> BSplines_Projdata(sino_3D_in, this_type);
	BasicCoordinate<3, BSplineType> & this_type ;
	const SegmentBySinogram<float> sino_3D_out = proj_data_out.get_segment_by_sinogram(0) ;
		
	for (float k=sino_3D_out.get_min_index() ; k<=sino_3D_out.get_max_index() ; ++k)	
		for (float j=sino_3D_out[k].get_min_index() ; j<=sino_3D_out[k].get_max_index() ; ++j)	
			for (float i=sino_3D_out[k][j].get_min_index() ; i<=sino_3D_out[k][j].get_max_index() ; ++i)	
			{				
				relative_positions[1]=k;
				relative_positions[2]=j;
				relative_positions[3]=i;
				sino_3D_out[k][j][i]=BSplines_Projdata(relative_positions) ; 
			}
	proj_data_out.set_segment(sino_3D_out);
	if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
		return Succeeded::no;
	break;
	return Succeeded::yes;
}
END_NAMESPACE_STIR
