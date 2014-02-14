//
//
/*!
  \file
  \ingroup symmetries
  \brief inline implementations for class stir::DataSymmetriesForBins_PET_CartesianGrid

  \author Mustapha Sadki
  \author Kris Thielemans
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

  Modification history:

  KT 30/05/2002 added possibility for reduced symmetry in view_num
*/
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/recon_buildblock/SymmetryOperations_PET_CartesianGrid.h"


START_NAMESPACE_STIR

#if 0
const DiscretisedDensityOnCartesianGrid<3,float> *    
DataSymmetriesForBins_PET_CartesianGrid::
cartesian_grid_info_ptr() const
{
  // we can use static_cast here, as the constructor already checked that it is type-safe
  return static_cast<const DiscretisedDensityOnCartesianGrid<3,float> *>
    (image_info_ptr.get());
}
#endif

float
DataSymmetriesForBins_PET_CartesianGrid::
get_num_planes_per_axial_pos(const int segment_num) const
{
  return static_cast<float>(num_planes_per_axial_pos[segment_num]);
}

float
DataSymmetriesForBins_PET_CartesianGrid::
get_num_planes_per_scanner_ring() const
{
  return static_cast<float>(num_planes_per_scanner_ring);
}

float 
DataSymmetriesForBins_PET_CartesianGrid::
get_axial_pos_to_z_offset(const int segment_num) const
{
  return axial_pos_to_z_offset[segment_num];
}	

int
DataSymmetriesForBins_PET_CartesianGrid::
find_transform_z(
		 const int segment_num, 
		 const int  axial_pos_num) const
{
  ProjDataInfoCylindrical* proj_data_info_cyl_ptr = 
    static_cast<ProjDataInfoCylindrical *>(proj_data_info_ptr.get());

  const float delta = proj_data_info_cyl_ptr->get_average_ring_difference(segment_num);

   
  // Find symmetric value in Z by 'mirroring' it around the centre z of the LOR:
  // Z+Q = 2*centre_of_LOR_in_image_coordinates == transform_z
  {
    // first compute it as floating point (although it has to be an int really)
    const float transform_z_float = (2*num_planes_per_axial_pos[segment_num]*(axial_pos_num) 
				     + num_planes_per_scanner_ring*delta
				     + 2*axial_pos_to_z_offset[segment_num]); 
    // now use rounding to be safe
    int transform_z = (int)floor(transform_z_float + 0.5);
    assert(fabs(transform_z-transform_z_float) < 10E-4);

    return transform_z;
  }
}

SymmetryOperation*
DataSymmetriesForBins_PET_CartesianGrid::
find_sym_op_bin0(   					 
                 int segment_num, 
                 int view_num, 
                 int axial_pos_num) const
{
  // note: if do_symmetry_shift_z==true, then basic axial_pos_num will be 0
  const int transform_z = 
	find_transform_z(abs(segment_num), 
	do_symmetry_shift_z ? 0 : axial_pos_num);

  const int z_shift = 
	do_symmetry_shift_z ?
	num_planes_per_axial_pos[segment_num]*axial_pos_num
	: 0;
  
  const int view180 = num_views;

  // TODO get rid of next 2 restrictions
  assert(!do_symmetry_180degrees_min_phi || view_num>=0);
  assert(!do_symmetry_180degrees_min_phi || view_num<num_views);

#ifndef NDEBUG
  // This variable is only used in assert() at the moment, so avoid compiler 
  // warning by defining it only when in debug mode
  const int view0   = 0;
#endif
  const int view135 = view180/4*3;
  const int view90  = view180/2;
  const int view45  = view180/4;

  if (  do_symmetry_90degrees_min_phi && view_num > view90 && view_num <= view135) {  //(90, 135 ]
    if ( !do_symmetry_swap_segment || segment_num >= 0)	
      return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(view180, axial_pos_num, z_shift);          
    else               
      return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq(view180, axial_pos_num, z_shift, transform_z);		   // seg < 0    				 			 
  } 
  else if ( do_symmetry_90degrees_min_phi && view_num > view45 && view_num <= view90  ) { // [ 45,  90] 		 
    if ( !do_symmetry_swap_segment || segment_num >= 0)  
      return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq(view180, axial_pos_num, z_shift, transform_z);  					 			  			 
    else
      return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx(view180, axial_pos_num, z_shift); // seg < 0   //KT????????????  different for view90, TODO				  
  }  
  else if( do_symmetry_180degrees_min_phi && view_num > view90/* && view_num <= view180 */){   // (135, 180) but (90,180) for reduced symmetry case
    if( !do_symmetry_swap_segment || segment_num >= 0)   
      return new SymmetryOperation_PET_CartesianGrid_swap_xmx_zq(view180, axial_pos_num, z_shift, transform_z);  
    else 	            
      return new SymmetryOperation_PET_CartesianGrid_swap_xmx(view180, axial_pos_num, z_shift);	  // seg < 0        				
  } 
  else 
  {
    assert( !do_symmetry_90degrees_min_phi || (view_num >= view0 && view_num <= view45));
    assert( !do_symmetry_180degrees_min_phi || (view_num >= view0 && view_num <= view90));
    if ( do_symmetry_swap_segment && segment_num < 0) 
      return new SymmetryOperation_PET_CartesianGrid_swap_zq(view180, axial_pos_num, z_shift, transform_z);                              
    else
    {
      if (z_shift==0)
       return new TrivialSymmetryOperation();
      else
        return new SymmetryOperation_PET_CartesianGrid_z_shift(axial_pos_num, z_shift);
    }
  }
}

// from symmetries
SymmetryOperation* 
DataSymmetriesForBins_PET_CartesianGrid::
find_sym_op_general_bin(   
                        int s, 
                        int segment_num, 
                        int view_num, 
                        int axial_pos_num) const
{ 
  // note: if do_symmetry_shift_z==true, then basic axial_pos_num will be 0
  const int transform_z = 
	find_transform_z(abs(segment_num), 
	do_symmetry_shift_z ? 0 : axial_pos_num);

  const int z_shift = 
	do_symmetry_shift_z ?
	num_planes_per_axial_pos[segment_num]*axial_pos_num
	: 0;
  
// TODO get rid of next 2 restrictions
  assert(!do_symmetry_180degrees_min_phi || view_num>=0);
  assert(!do_symmetry_180degrees_min_phi || view_num<num_views);

  const int view180 = num_views;
#ifndef NDEBUG
  // This variable is only used in assert() at the moment, so avoid compiler 
  // warning by defining it only when in debug mode
  const int view0   = 0;
#endif

  const int view135 = view180/4*3;
  const int view90  = view180/2;
  const int view45  = view180/4;
  
  
  if (  do_symmetry_90degrees_min_phi && view_num > view90 && view_num <= view135) {  //(90, 135 ]
    if ( !do_symmetry_swap_segment || segment_num > 0) {	 // pos_plus90		 
      if ( !do_symmetry_swap_s || s > 0 ) 
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(view180, axial_pos_num, z_shift);           				    			   
      else 
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq(view180, axial_pos_num, z_shift, transform_z); // s < 0  					 
    }  
    else // neg_plus90
      /////
      if ( segment_num < 0 )	{   
        if ( !do_symmetry_swap_s || s > 0 )  
          return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq(view180, axial_pos_num, z_shift, transform_z);   
        else     
          return new SymmetryOperation_PET_CartesianGrid_swap_xy_ymx(view180, axial_pos_num, z_shift);	     						  
      }   
      else { // segment_num == 0 							      
        if ( !do_symmetry_swap_s || s > 0 ) 
          return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(view180, axial_pos_num, z_shift); 
        else         
          return new SymmetryOperation_PET_CartesianGrid_swap_xy_ymx(view180, axial_pos_num, z_shift);			 
      }			   			    					  
  }    
  else   if ( do_symmetry_90degrees_min_phi && view_num > view45 && view_num <= view90  )  // [ 45,  90] 
  {		   
    if ( !do_symmetry_swap_segment || segment_num > 0){  
      if ( !do_symmetry_swap_s || s > 0 ) 	   
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq(view180, axial_pos_num, z_shift, transform_z); 					 				  
      else             
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx(view180, axial_pos_num, z_shift);	 			   
    }
    else if ( segment_num < 0 ) { // {//101   segment_num < 0
      if ( !do_symmetry_swap_s || s > 0 )     
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx(view180, axial_pos_num, z_shift);			   
      else    
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq(view180, axial_pos_num, z_shift, transform_z);     
      
    } 
    else // segment_num == 0
    {
      if ( !do_symmetry_swap_s || s > 0 ) 
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx(view180, axial_pos_num, z_shift);		
      else           
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx(view180, axial_pos_num, z_shift);				       
    }
  }  
  else if( do_symmetry_180degrees_min_phi && view_num > view90/* && view_num <= view180 */)   // (135, 180) but (90,180) for reduced symmetry case    
  {
    if( !do_symmetry_swap_segment || segment_num > 0){				    
      if ( !do_symmetry_swap_s || s > 0 )     
        return new SymmetryOperation_PET_CartesianGrid_swap_xmx_zq(view180, axial_pos_num, z_shift, transform_z);  					
      else             
        return new SymmetryOperation_PET_CartesianGrid_swap_ymy(view180, axial_pos_num, z_shift);     //  s <= 0  						
    }
    else //if ( segment_num < 0 )
    {// segment_num <= 0
      if ( !do_symmetry_swap_s || s > 0 ) 
        return new SymmetryOperation_PET_CartesianGrid_swap_xmx(view180, axial_pos_num, z_shift);    				    
      else 	         
        return new SymmetryOperation_PET_CartesianGrid_swap_ymy_zq(view180, axial_pos_num, z_shift, transform_z);	 					     				   
    }// segment_num == 0
    // /*else{   if ( !do_symmetry_swap_s || s > 0 ) return new SymmetryOperation_PET_CartesianGrid_swap_xmx();	else 	return new SymmetryOperation_PET_CartesianGrid_swap_ymy(view180, axial_pos_num, z_shift);}*/
  }  
  else 
  {    
    assert( !do_symmetry_90degrees_min_phi || (view_num >= view0 && view_num <= view45));
    assert( !do_symmetry_180degrees_min_phi || (view_num >= view0 && view_num <= view90));
    if ( !do_symmetry_swap_segment || segment_num > 0) 
    {   
      if ( do_symmetry_swap_s && s < 0) 
        return new SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq(view180, axial_pos_num, z_shift, transform_z);   									    						   
      else
      {
        if (z_shift==0)
          return new TrivialSymmetryOperation();
        else
          return new SymmetryOperation_PET_CartesianGrid_z_shift(axial_pos_num, z_shift);
      }
    }
    else 
      if ( segment_num < 0 ) 	
      {
        /*KT if ( s == 0)   					 
          return new SymmetryOperation_PET_CartesianGrid_swap_zq(view180, axial_pos_num, z_shift, transform_z); 										
        else*/  
          if ( do_symmetry_swap_s && s < 0) 
            return new SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy(view180, axial_pos_num, z_shift);  
          else        
            return new SymmetryOperation_PET_CartesianGrid_swap_zq(view180, axial_pos_num, z_shift, transform_z);   // s > 0  						      						      						                         					
      }  
      else // segment_num = 0 
      {
        if ( do_symmetry_swap_s && s < 0) return new SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy(view180, axial_pos_num, z_shift); 
        else
        {
          if (z_shift==0)
            return new TrivialSymmetryOperation();
          else
            return new SymmetryOperation_PET_CartesianGrid_z_shift(axial_pos_num, z_shift);
        }
      }
  }			
}


bool  
DataSymmetriesForBins_PET_CartesianGrid::
find_basic_view_segment_numbers(ViewSegmentNumbers& v_s) const 
{
   bool change=false;
 // TODO get rid of next 2 restrictions
  assert(!do_symmetry_180degrees_min_phi || v_s.view_num()>=0);
  assert(!do_symmetry_180degrees_min_phi || v_s.view_num()<num_views);

  //const int view0=  0;
   const int view90  =  num_views>>1;   
   const int view45  =  view90>>1;
   const int view135 =  view90+view45;

   if ( do_symmetry_swap_segment && v_s.segment_num() < 0    )  { v_s.segment_num() = -v_s.segment_num(); change=true;}

   if (do_symmetry_90degrees_min_phi)
   {
     //if ( v_s.view_num() == num_views )    v_s.view_num() =0;	// KT 30/05/2002 disabled as it should never happen
     //else 
     if (v_s.view_num() >= view135)  
     { v_s.view_num() = num_views - v_s.view_num(); return true; }
     else if (v_s.view_num()  >=  view90  ) 
     { v_s.view_num() =  v_s.view_num() - view90;  return true; }	
     else if (v_s.view_num()  > view45 ) 
     { v_s.view_num()  = view90 - v_s.view_num() ;  return true; }
   }
   else if (do_symmetry_180degrees_min_phi)
   {
     if (v_s.view_num() > view90  ) 
     { v_s.view_num() =  num_views - v_s.view_num();  return true; }
   }
   
  return change;
}

bool  
DataSymmetriesForBins_PET_CartesianGrid::
find_basic_bin(int &segment_num, int &view_num, int &axial_pos_num, int &tangential_pos_num) const 
{
  ViewSegmentNumbers v_s(view_num, segment_num);

  bool change=find_basic_view_segment_numbers(v_s);

  view_num = v_s.view_num();
  segment_num = v_s.segment_num();

  if ( do_symmetry_swap_s && tangential_pos_num < 0      )  { tangential_pos_num   = - tangential_pos_num ; change=true;};
  if ( do_symmetry_shift_z && axial_pos_num != 0    )  { axial_pos_num  =  0;     change = true; }   
  
  return change;
}

bool  
DataSymmetriesForBins_PET_CartesianGrid::
find_basic_bin(Bin& b) const 
{
  return 
    find_basic_bin(b.segment_num(), b.view_num(), b.axial_pos_num(), b.tangential_pos_num());
}


// TODO, optimise
auto_ptr<SymmetryOperation>
DataSymmetriesForBins_PET_CartesianGrid::
  find_symmetry_operation_from_basic_bin(Bin& b) const
{
  auto_ptr<SymmetryOperation> 
    sym_op(
      (b.tangential_pos_num()==0) ?
        find_sym_op_bin0(b.segment_num(), b.view_num(), b.axial_pos_num()) :
        find_sym_op_general_bin(b.tangential_pos_num(), b.segment_num(), b.view_num(), b.axial_pos_num())
      ); 
  find_basic_bin(b);
  return sym_op;
}


int
DataSymmetriesForBins_PET_CartesianGrid::
num_related_view_segment_numbers(const ViewSegmentNumbers& vs) const
{      
  int num = do_symmetry_180degrees_min_phi  && (vs.view_num() % (num_views/2)) != 0 ? 2 : 1;
  if (do_symmetry_90degrees_min_phi && (vs.view_num() % (num_views/2)) != num_views/4)
    num *= 2;
  if (do_symmetry_swap_segment && vs.segment_num() != 0)
    num *= 2;
 return num;
}


int
DataSymmetriesForBins_PET_CartesianGrid::
num_related_bins(const Bin& b) const
{
  int num = do_symmetry_180degrees_min_phi  && (b.view_num() % (num_views/2)) != 0 ? 2 : 1;
  if (do_symmetry_90degrees_min_phi && (b.view_num() % (num_views/2)) != num_views/4)
    num *= 2;
  if (do_symmetry_swap_segment && b.segment_num() != 0)
    num *= 2;

  if (do_symmetry_swap_s && b.tangential_pos_num() != 0)
    num *= 2;
  
  if (do_symmetry_shift_z)
    num *= proj_data_info_ptr->get_num_axial_poss(b.segment_num());
  return num;
}

void
DataSymmetriesForBins_PET_CartesianGrid::
get_related_bins_factorised(vector<AxTangPosNumbers>& ax_tang_poss, const Bin& b,
                            const int min_axial_pos_num, const int max_axial_pos_num,
                            const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  for (int axial_pos_num=do_symmetry_shift_z?min_axial_pos_num:b.axial_pos_num();
       axial_pos_num <= (do_symmetry_shift_z?max_axial_pos_num:b.axial_pos_num());
       ++axial_pos_num)
  {
     if (b.tangential_pos_num() >= min_tangential_pos_num &&
         b.tangential_pos_num() <= max_tangential_pos_num)
        ax_tang_poss.push_back(AxTangPosNumbers(axial_pos_num, b.tangential_pos_num()));
     if (do_symmetry_swap_s && b.tangential_pos_num()!=0 &&
         -b.tangential_pos_num() >= min_tangential_pos_num &&
         -b.tangential_pos_num() <= max_tangential_pos_num)
        ax_tang_poss.push_back(AxTangPosNumbers(axial_pos_num, -b.tangential_pos_num()));
    }
}
    
void
DataSymmetriesForBins_PET_CartesianGrid::
get_related_view_segment_numbers(vector<ViewSegmentNumbers>& rel_vs, const ViewSegmentNumbers& vs) const
{
#ifndef NDEBUG
  {
    ViewSegmentNumbers vstest=vs;
    assert(find_basic_view_segment_numbers(vstest)==false);
  }
#endif

  const int segment_num = vs.segment_num();
  const int view_num = vs.view_num();

  const bool symz = 
    do_symmetry_swap_segment && (segment_num != 0);

  rel_vs.reserve(num_related_view_segment_numbers(vs));
  rel_vs.resize(0);

  rel_vs.push_back(ViewSegmentNumbers(view_num,segment_num));

  if (symz)
    rel_vs.push_back(ViewSegmentNumbers(view_num,-segment_num));

  if (do_symmetry_180degrees_min_phi && do_symmetry_90degrees_min_phi && (view_num % (num_views/2)) != num_views/4)
  {
    const int related_view_num = 
      view_num < num_views/2 ?
      view_num + num_views/2 :
      view_num - num_views/2;
    rel_vs.push_back(ViewSegmentNumbers( related_view_num,segment_num));
    if (symz)
      rel_vs.push_back(ViewSegmentNumbers( related_view_num,-segment_num));
  }

  if (do_symmetry_180degrees_min_phi && (view_num % (num_views/2)) != 0)
  {
    rel_vs.push_back(ViewSegmentNumbers( num_views - view_num,segment_num));
    if (symz)
      rel_vs.push_back(ViewSegmentNumbers( num_views - view_num,-segment_num));
  }
  if (do_symmetry_90degrees_min_phi && (view_num % (num_views/4)) != 0)
  {
    // use trick to get related_view_num between 0 and num_views:
    // use modulo num_views (but add num_views first to ensure positivity)
    const int related_view_num = 
      (num_views/2 - view_num + num_views) % num_views;
    rel_vs.push_back(ViewSegmentNumbers( related_view_num,segment_num));
    if (symz)
      rel_vs.push_back(ViewSegmentNumbers( related_view_num,-segment_num));
  }


  assert(rel_vs.size() == 
         static_cast<unsigned>(num_related_view_segment_numbers(vs)));
}

END_NAMESPACE_STIR
