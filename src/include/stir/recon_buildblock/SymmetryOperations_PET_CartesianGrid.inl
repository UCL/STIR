//
//
/*!

  \file
  \ingroup symmetries

  \brief Implementation of all symmetry classes for PET scanners and cartesian images

  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
/* History:
   KT 07/10/2001:
   made sure that the resulting view_num is in the allowed range
   use that an LOR at {theta,phi,m,s} is equal to the one at {-theta, phi+Pi,m, -s}
   TODO, didn't do it for the _zq symmetries, as I don't need it there yet
   */

#include "stir/BasicCoordinate.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Bin.h"

START_NAMESPACE_STIR

void 
SymmetryOperation_PET_CartesianGrid_z_shift::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
}

void 
SymmetryOperation_PET_CartesianGrid_z_shift::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
}
 
void SymmetryOperation_PET_CartesianGrid_z_shift::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{			
  c[1] += z_shift;
}

//////////////////////////////////////////

void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.view_num() = view180 - b.view_num();
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num() = view180 - vs.view_num();
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 
void SymmetryOperation_PET_CartesianGrid_swap_xmx_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{			
  c[3] = -c[3];		 
  c[1] = q - c[1] + z_shift;
}

//////////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.segment_num() *= -1;
  b.view_num() += view180/2;
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}

void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num() += view180/2;
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 

void SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{
  const int tmp = c[3];
  c[3] = -c[2];
  c[2] = tmp;
  c[1] = q - c[1] + z_shift;
}

///////////////////////////////////////

void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.view_num() = view180/2 - b.view_num();
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  

}
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num() = view180/2 - vs.view_num();
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 
void SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{    
  const int tmp = c[3];
  c[3] = c[2];
  c[2] = tmp;
  c[1] = q - c[1] + z_shift;
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  if (b.view_num() < view180/2)
  {
    b.view_num() += view180/2;
  }
  else
  {
    b.segment_num() *= -1;
    b.view_num() -= view180/2;
    b.tangential_pos_num() *= -1;
  }
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}

void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  if (vs.view_num() < view180/2)
  {
    vs.view_num() += view180/2;
  }
  else
  {
    vs.segment_num() *= -1;
    vs.view_num() -= view180/2;
  }
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 
void SymmetryOperation_PET_CartesianGrid_swap_xmy_yx::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{   
  const int tmp = c[3];
  c[3] = -c[2];
  c[2] = tmp;
  c[1] += z_shift;
}

///////////////////////////////////////
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  if (b.view_num() <= view180/2)
  {
    b.segment_num() *= -1;
    b.view_num() = view180/2 - b.view_num();
  }
  else
  {
    b.view_num() = 3*view180/2 - b.view_num();
    b.tangential_pos_num() *= -1;
  }
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  if (vs.view_num() <= view180/2)
  {
    vs.segment_num() *= -1;
    vs.view_num() = view180/2 - vs.view_num();
  }
  else
  {
    vs.view_num() = 3*view180/2 - vs.view_num();
  }
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 

void SymmetryOperation_PET_CartesianGrid_swap_xy_yx::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{  
  const int tmp = c[3];
  c[3] = c[2];
  c[2] = tmp;		
  c[1] += z_shift;
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xmx::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  if (b.view_num()!=0)
  {
    b.segment_num() *= -1;
    b.view_num() = view180 - b.view_num();
  }
  else
  {
    b.tangential_pos_num() *= -1;
  }
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}

void 
SymmetryOperation_PET_CartesianGrid_swap_xmx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  if (vs.view_num()!=0)
  {
    vs.segment_num() *= -1;
    vs.view_num() = view180 - vs.view_num();
  }
  else
  {}
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 

void SymmetryOperation_PET_CartesianGrid_swap_xmx::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{   
  c[3] = -c[3];		
  c[1] += z_shift;
} 		         

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_ymy::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  if (b.view_num()!=0)
  {
    b.view_num() = view180 - b.view_num();
    b.tangential_pos_num() *= -1;
  }
  else
  {
    b.segment_num() *= -1;
  }
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}
void 
SymmetryOperation_PET_CartesianGrid_swap_ymy::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  if (vs.view_num()!=0)
  {
    vs.view_num() = view180 - vs.view_num();
  }
  else
  {
    vs.segment_num() *= -1;
  }
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 

void SymmetryOperation_PET_CartesianGrid_swap_ymy::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{   		 	    
  c[2] = -c[2];
  c[1] += z_shift;
} 		         

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.segment_num() *= -1;
}

void 
SymmetryOperation_PET_CartesianGrid_swap_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
}
 

void SymmetryOperation_PET_CartesianGrid_swap_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{   		 
  c[1] = q - c[1] + z_shift;       
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.tangential_pos_num() *= -1;
}

void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  
}
 

void SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{ 
  c[3] = -c[3];
  c[2] = -c[2];	
  c[1] = q - c[1] + z_shift;	 		  	 		 
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  if (b.view_num() < view180/2)
  {
    b.view_num() += view180/2;
    b.tangential_pos_num() *= -1;
  }
  else
  {
    b.segment_num() *= -1;
    b.view_num() -= view180/2;
  }
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  

}

void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  if (vs.view_num() < view180/2)
  {
    vs.view_num() += view180/2;
  }
  else
  {
    vs.segment_num() *= -1;
    vs.view_num() -= view180/2;
  }
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 

void SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{  
  const int tmp = c[3];
  c[3] = c[2];
  c[2] = -tmp;		
  c[1] = q - c[1] + z_shift;	     	 
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  if (b.view_num() < view180/2)
  {
    b.segment_num() *= -1;
    b.view_num() += view180/2;
    b.tangential_pos_num() *= -1;
  }
  else
  {
    b.view_num() -= view180/2;
  }
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  if (vs.view_num() < view180/2)
  {
    vs.segment_num() *= -1;
    vs.view_num() += view180/2;
  }
  else
  {
    vs.view_num() -= view180/2;
  }
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 
void SymmetryOperation_PET_CartesianGrid_swap_xy_ymx::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{
  const int tmp = c[3];
  c[3] = c[2];	
  c[2] = -tmp;		
  c[1] += z_shift;
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  if (b.view_num() <= view180/2)
  {
    b.view_num() = view180/2 - b.view_num();
    b.tangential_pos_num() *= -1;
  }
  else
  {
    b.segment_num() *= -1;
    b.view_num() = 3*view180/2 - b.view_num();
  }
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}

void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  if (vs.view_num() <= view180/2)
  {
    vs.view_num() = view180/2 - vs.view_num();
  }
  else
  {
    vs.segment_num() *= -1;
    vs.view_num() = 3*view180/2 - vs.view_num();
  }
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 
void SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{  
  const int tmp = c[3];
  c[3] = -c[2];	
  c[2] = -tmp;			 
  c[1] += z_shift;
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_ymy_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.segment_num() *= -1;
  b.view_num() = view180 - b.view_num();
  b.tangential_pos_num() *= -1;
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}

void 
SymmetryOperation_PET_CartesianGrid_swap_ymy_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num()= view180 - vs.view_num();
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 

void SymmetryOperation_PET_CartesianGrid_swap_ymy_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{  		   
  c[2] = -c[2];		
  c[1] = q - c[1] + z_shift;     	
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.segment_num() *= -1;
  b.tangential_pos_num() *= -1;
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
}
 
void SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{    
  c[3] = -c[3];	
  c[2] = -c[2];	
  c[1] += z_shift;
}

///////////////////////////////////////


void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq::
    transform_bin_coordinates(Bin& b) const
{
  b.axial_pos_num() += axial_pos_shift;
  b.segment_num() *= -1;
  b.view_num() = view180/2 - b.view_num();
  b.tangential_pos_num() *= -1;
  assert(0<=b.view_num());
  assert(b.view_num()<view180);  
}

void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num() = view180/2 - vs.view_num();
  assert(0<=vs.view_num());
  assert(vs.view_num()<view180);  
}
 
void SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{
  const int tmp = c[3];
  c[3] = -c[2];
  c[2] = -tmp;		
  c[1] = q - c[1] + z_shift;
}

END_NAMESPACE_STIR
