//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief inline implementations for class DataSymmetriesForBins_PET_CartesianGrid

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
#include "ProjDataInfoCylindrical.h"
#include "recon_buildblock/SymmetryOperations_PET_CartesianGrid.h"


START_NAMESPACE_TOMO

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
    // KT 27/09/2000 pass 0 instead of axial_pos_num
  const int transform_z = find_transform_z(abs(segment_num), 0);

  const int z_shift = num_planes_per_axial_pos[segment_num]*axial_pos_num;
  
  const int view180 = num_views;
#ifndef NDEBUG
  // This variable is only used in assert() at the moment, so avoid compiler 
  // warning by defining it only when in debug mode
  const int view0   = 0;
#endif
  const int view135 = view180/4*3;
  const int view90  = view180/2;
  const int view45  = view180/4;

  if (  view_num > view90 && view_num <= view135 ) {  //[90, 135 ]		
    if ( segment_num >= 0)	
      return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(view180, axial_pos_num, z_shift);          
    else               
      return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq(view180, axial_pos_num, z_shift, transform_z);		   // seg < 0    				 			 
  } 
  else if ( view_num > view45 && view_num <= view90  ) { // [ 45,  90] 		 
    if ( segment_num >= 0)  
      return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq(view180, axial_pos_num, z_shift, transform_z);  					 			  			 
    else
      return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx(view180, axial_pos_num, z_shift); // seg < 0   //KT????????????  different for view90, TODO				  
  }  
  else if( view_num > view135 && view_num <= view180 ){   // 135 180 
    if( segment_num >= 0)   
      return new SymmetryOperation_PET_CartesianGrid_swap_xmx_zq(view180, axial_pos_num, z_shift, transform_z);  
    else 	            
      return new SymmetryOperation_PET_CartesianGrid_swap_xmx(view180, axial_pos_num, z_shift);	  // seg < 0        				
  } 
  else 
  {
    assert( view_num >= view0 && view_num <= view45);
    if ( segment_num < 0) 
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
  // KT 27/09/2000 pass 0 instead of axial_pos_num
  const int transform_z = find_transform_z(abs(segment_num), 0);

  const int z_shift = num_planes_per_axial_pos[segment_num]*axial_pos_num;
  
  const int view180 = num_views;
#ifndef NDEBUG
  // This variable is only used in assert() at the moment, so avoid compiler 
  // warning by defining it only when in debug mode
  const int view0   = 0;
#endif

  const int view135 = view180/4*3;
  const int view90  = view180/2;
  const int view45  = view180/4;
  
  
  if (  view_num > view90 && view_num <= view135 ) {  //[90, 135 ]		
    if ( segment_num >  0) {	 // pos_plus90		 
      if ( s > 0 ) 
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(view180, axial_pos_num, z_shift);           				    			   
      else 
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq(view180, axial_pos_num, z_shift, transform_z); // s < 0  					 
    }  
    else // neg_plus90
      /////
      if ( segment_num < 0 )	{   
        if ( s > 0 )  
          return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq(view180, axial_pos_num, z_shift, transform_z);   
        else     
          return new SymmetryOperation_PET_CartesianGrid_swap_xy_ymx(view180, axial_pos_num, z_shift);	     						  
      }   
      else { // segment_num == 0 							      
        if ( s > 0 ) 
          return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(view180, axial_pos_num, z_shift); 
        else         
          return new SymmetryOperation_PET_CartesianGrid_swap_xy_ymx(view180, axial_pos_num, z_shift);			 
      }			   			    					  
  }    
  else   if ( view_num > view45 && view_num <= view90  )  // [ 45,  90] 
  {		   
    if ( segment_num > 0){  
      if ( s > 0 ) 	   
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq(view180, axial_pos_num, z_shift, transform_z); 					 				  
      else             
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx(view180, axial_pos_num, z_shift);	 			   
    }
    else if ( segment_num < 0 ) { // {//101   segment_num < 0
      if ( s > 0 )     
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx(view180, axial_pos_num, z_shift);			   
      else    
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq(view180, axial_pos_num, z_shift, transform_z);     
      
    } 
    else // segment_num == 0
    {
      if ( s > 0 ) 
        return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx(view180, axial_pos_num, z_shift);		
      else           
        return new SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx(view180, axial_pos_num, z_shift);				       
    }
  }  
  else  if( view_num > view135 && view_num <= view180 )   // 135 180 
  {
    if( segment_num > 0){				    
      if ( s > 0 )     
        return new SymmetryOperation_PET_CartesianGrid_swap_xmx_zq(view180, axial_pos_num, z_shift, transform_z);  					
      else             
        return new SymmetryOperation_PET_CartesianGrid_swap_ymy(view180, axial_pos_num, z_shift);     //  s <= 0  						
    }
    else //if ( segment_num < 0 )
    {// segment_num <= 0
      if ( s > 0 ) 
        return new SymmetryOperation_PET_CartesianGrid_swap_xmx(view180, axial_pos_num, z_shift);    				    
      else 	         
        return new SymmetryOperation_PET_CartesianGrid_swap_ymy_zq(view180, axial_pos_num, z_shift, transform_z);	 					     				   
    }// segment_num == 0
    // /*else{   if ( s > 0 ) return new SymmetryOperation_PET_CartesianGrid_swap_xmx();	else 	return new SymmetryOperation_PET_CartesianGrid_swap_ymy(view180, axial_pos_num, z_shift);}*/
  }  
  else 
  {    
    assert( view_num >= view0 && view_num <= view45 );
    if ( segment_num > 0) 
    {   
      if ( s < 0) 
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
          if ( s < 0) 
            return new SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy(view180, axial_pos_num, z_shift);  
          else        
            return new SymmetryOperation_PET_CartesianGrid_swap_zq(view180, axial_pos_num, z_shift, transform_z);   // s > 0  						      						      						                         					
      }  
      else // segment_num = 0 
      {
        if ( s < 0) return new SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy(view180, axial_pos_num, z_shift); 
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
   //const int view0=  0;
   const int view90  =  num_views>>1;   
   const int view45  =  view90>>1;
   const int view135 =  view90+view45;

   if ( v_s.segment_num() < 0    )  { v_s.segment_num() = -v_s.segment_num(); change=true;};	   
   if ( v_s.view_num() > view45)  { change=true;  }
  
   if ( v_s.view_num() == num_views )    v_s.view_num() =0;	
   else if ( v_s.view_num()  >=  view135)  v_s.view_num() = num_views - v_s.view_num(); 		
   else if ( v_s.view_num()  >=  view90  )    v_s.view_num() =  v_s.view_num() - view90;		
   else if ( v_s.view_num()  > view45 ) v_s.view_num()  = view90 - v_s.view_num() ;
   
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

  if ( tangential_pos_num < 0      )  { tangential_pos_num   = - tangential_pos_num ; change=true;};
  if ( axial_pos_num != 0    )  { axial_pos_num  =  0;     change = true; }   
  
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
  find_symmetry_operation_to_basic_bin(Bin& b) const
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
  int num = 2;
  if ((vs.view_num() % (num_views/4)) != 0)
    num *= 2;
  if (vs.segment_num() != 0)
    num *= 2;
 return num;
}


int
DataSymmetriesForBins_PET_CartesianGrid::
num_related_bins(const Bin& b) const
{
  int num = 2;
  if ((b.view_num() % (num_views/4)) != 0)
    num *= 2;
  if (b.segment_num() != 0)
    num *= 2;

  if (b.tangential_pos_num() != 0)
    num *= 2;
  
  num *= proj_data_info_ptr->get_num_axial_poss(b.segment_num());
  return num;
}

void
DataSymmetriesForBins_PET_CartesianGrid::
get_related_bins_factorised(vector<AxTangPosNumbers>& ax_tang_poss, const Bin& b,
                            const int min_axial_pos_num, const int max_axial_pos_num,
                            const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  for (int axial_pos_num=min_axial_pos_num;
       axial_pos_num <= max_axial_pos_num;
       ++axial_pos_num)
  {
     if (b.tangential_pos_num() >= min_tangential_pos_num &&
         b.tangential_pos_num() <= max_tangential_pos_num)
        ax_tang_poss.push_back(AxTangPosNumbers(axial_pos_num, b.tangential_pos_num()));
     if (b.tangential_pos_num()!=0 &&
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

  const bool symviewplus90 = true;
  const bool sym90minview = 
    (view_num % (num_views/4)) != 0;
  const bool symz = 
    (segment_num != 0);

  rel_vs.reserve(num_related_view_segment_numbers(vs));
  rel_vs.resize(0);

  rel_vs.push_back(ViewSegmentNumbers(view_num,segment_num));

  if (symz)
    rel_vs.push_back(ViewSegmentNumbers(view_num,-segment_num));

  if (symviewplus90)
  {
    const int related_view_num = 
      view_num < num_views/2 ?
      view_num + num_views/2 :
      view_num - num_views/2;
    rel_vs.push_back(ViewSegmentNumbers( related_view_num,segment_num));
    if (symz)
      rel_vs.push_back(ViewSegmentNumbers( related_view_num,-segment_num));
  }

  if (symviewplus90 && sym90minview)
  {
    rel_vs.push_back(ViewSegmentNumbers( num_views - view_num,segment_num));
    if (symz)
      rel_vs.push_back(ViewSegmentNumbers( num_views - view_num,-segment_num));
  }
  if (sym90minview)
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

END_NAMESPACE_TOMO
