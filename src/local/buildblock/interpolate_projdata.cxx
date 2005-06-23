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
#include "stir/BasicCoordinate.h"
#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"
#include "local/stir/interpolate_projdata.h"
#include "stir/Sinogram.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"



START_NAMESPACE_STIR

using namespace BSpline;

/*Succeeded 
interpolate_projdata(ProjData& proj_data_out,
					 const ProjData& proj_data_in, const BSplineType these_types)
{
	BasicCoordinate<3, BSplineType> these_types_3; these_types_3[1]=these_types_3[2]=these_types_3[3]=these_types;
	interpolate_projdata(proj_data_out,proj_data_in,these_types_3);
	return Succeeded::yes;
}*/

Succeeded 
interpolate_projdata(ProjData& proj_data_out,
					 const ProjData& proj_data_in,
					 const BasicCoordinate<3, BSplineType> & these_types)
{
	/*const ProjDataInfo * const proj_data_in_info_ptr =
		dynamic_cast<ProjDataInfo const * >
		(proj_data_in.get_proj_data_info_ptr());
	const ProjDataInfo * const proj_data_out_info_ptr =
		dynamic_cast<ProjDataInfo const * >
		(proj_data_out.get_proj_data_info_ptr());*/
	
	const SegmentBySinogram<float> sino_3D_in = proj_data_in.get_segment_by_sinogram(0) ;
	SegmentBySinogram<float> sino_3D_out = proj_data_out.get_empty_segment_by_sinogram(0) ;
	BSpline::BSplinesRegularGrid<3, float, float> BSplines_Projdata(sino_3D_in, these_types);
	BasicCoordinate<3, double> relative_positions, min_in,  max_in, length_in, 
		min_out, max_out, length_out, step  ;


	int kmin=sino_3D_in.get_min_index(),
	jmin=sino_3D_in[kmin].get_min_index(),
	imin=sino_3D_in[kmin][jmin].get_min_index(),
	kmax=sino_3D_in.get_max_index(),
	jmax=sino_3D_in[kmax].get_max_index(),
	imax=sino_3D_in[kmax][jmax].get_max_index();
/*	for (int k=sino_3D_in.get_min_index(), kmin=k; k<=sino_3D_in.get_max_index(); ++k)
		for (int j=sino_3D_in.get_min_index(), jmin=j; j<=sino_3D_in[k].get_max_index(); ++j)
			for (int i=sino_3D_in.get_min_index(), imin=i; i<=sino_3D_in[k][j].get_max_index(); ++i)
			{
				kmin=min(k,kmin); jmin=min(j,jmin); imin=min(i,imin);				
				kmax=max(k,kmax); jmax=max(j,jmax); imax=max(i,imax);				
			}
*/	min_in[1]=static_cast<double>(kmin);
	min_in[2]=static_cast<double>(jmin);
	min_in[3]=static_cast<double>(imin);
	max_in[1]=static_cast<double>(kmax);
	max_in[2]=static_cast<double>(jmax);
	max_in[3]=static_cast<double>(imax);

	kmin=sino_3D_out.get_min_index();
	jmin=sino_3D_out[kmin].get_min_index();
	imin=sino_3D_out[kmin][jmin].get_min_index();
	kmax=sino_3D_out.get_max_index();
	jmax=sino_3D_out[kmax].get_max_index();
	imax=sino_3D_out[kmax][jmax].get_max_index();;
/*	for (int k=sino_3D_out.get_min_index(), kmin=k; k<=sino_3D_out.get_max_index(); ++k)
		for (int j=sino_3D_out.get_min_index(), jmin=j; j<=sino_3D_out[k].get_max_index(); ++j)
			for (int i=sino_3D_out.get_min_index(), imin=i; i<=sino_3D_out[k][j].get_max_index(); ++i)
			{
				kmin=min(k,kmin); jmin=min(j,jmin); imin=min(i,imin);				
				kmax=max(k,kmax); jmax=max(j,jmax); imax=max(i,imax);				
			}
*/  min_out[1]=static_cast<double>(kmin);
	min_out[2]=static_cast<double>(jmin);
	min_out[3]=static_cast<double>(imin);
	max_out[1]=static_cast<double>(kmax);
	max_out[2]=static_cast<double>(jmax);
	max_out[3]=static_cast<double>(imax);
	
	length_in = (max_in-min_in+1.) ;
	length_out = (max_out-min_out+1.) ;
	step = length_in/(length_out +1.);
	step[2] -= .5/(length_out[2] +1.);

	//zoom = length_out/length_in OR 1/step ???;
	
	int k_out=kmin, j_out=jmin, i_out=imin;
			
	for (double k_in=min_in[1]-0.5+step[1] ; 
	         k_out<=max_out[1] && k_in<=max_in[1]+0.5-step[1] ; 
		   ++k_out, k_in+=step[1])	
		for (double j_in=min_in[2] ; 
		         j_out<=max_out[2] && j_in<=max_in[2]+0.5-step[2] ; 
		       ++j_out, j_in+=step[2])	
			for (double i_in=min_in[3]-0.5+step[3] ; 
			         i_out<=max_out[3] && i_in<=max_in[3]+0.5-step[3] ; 
				   ++i_out, i_in+=step[3])	
			{				
				relative_positions[1]=k_in;
				relative_positions[2]=j_in;
				relative_positions[3]=i_in;
				sino_3D_out[k_out][j_out][i_out]=
					BSplines_Projdata(relative_positions) ; 
			}
	proj_data_out.set_segment(sino_3D_out);
	if (proj_data_out.set_segment(sino_3D_out) == Succeeded::no)
		return Succeeded::no;

	return Succeeded::yes;
}
END_NAMESPACE_STIR
