//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of all symmetry classes for PET scanners and cartesian images

  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "BasicCoordinate.h"
#include "ViewSegmentNumbers.h"
#include "Bin.h"

START_NAMESPACE_TOMO

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
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num() = view180 - vs.view_num();
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
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num() += view180/2;
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
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num() = view180/2 - vs.view_num();
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
  b.view_num() += view180/2;
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_yx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num() += view180/2;
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
  b.segment_num() *= -1;
  b.view_num() = view180/2 - b.view_num();
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_yx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num() = view180/2 - vs.view_num();
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
  b.segment_num() *= -1;
  b.view_num() = view180 - b.view_num();
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num() = view180 - vs.view_num();
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
  b.view_num() = view180 - b.view_num();
  b.tangential_pos_num() *= -1;
}
void 
SymmetryOperation_PET_CartesianGrid_swap_ymy::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num() = view180 - vs.view_num();
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
  b.view_num() += view180/2;
  b.tangential_pos_num() *= -1;
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num() += view180/2;
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
  b.segment_num() *= -1;
  b.view_num() += view180/2;
  b.tangential_pos_num() *= -1;
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xy_ymx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num()+= view180/2;
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
  b.view_num() = view180/2 - b.view_num();
  b.tangential_pos_num() *= -1;
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.view_num()= view180/2 - vs.view_num();
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
}
void 
SymmetryOperation_PET_CartesianGrid_swap_ymy_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num()= view180 - vs.view_num();
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
}
void 
SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq::
    transform_view_segment_indices(ViewSegmentNumbers& vs) const
{
  vs.segment_num() *= -1;
  vs.view_num() = view180/2 - vs.view_num();
}
 

void SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq::transform_image_coordinates(BasicCoordinate<3,int>&c) const
{
  const int tmp = c[3];
  c[3] = -c[2];
  c[2] = -tmp;		
  c[1] = q - c[1] + z_shift;
}
END_NAMESPACE_TOMO
