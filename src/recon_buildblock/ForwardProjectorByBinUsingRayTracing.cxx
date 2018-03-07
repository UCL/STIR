//
//
/*!
  \file
  \ingroup projection

  \brief Implementations of non-static methods of stir::ForwardProjectorByBinUsingRayTracing.

  \author Kris Thielemans
  \author Claire Labbe
  \author Damiano Belluzzo
  \author (based originally on C code by Matthias Egger)
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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
/* 
  History:
  * Matthias Egger: C version 

  * PARAPET project : C++ version

  * KT : conversion to class and 'new design'

  * KT 20/06/2001 :
    - allow even-sized images and projdata
    - correctly handle arbitrary min,max_tang_pos_num
    - release restriction to arc-corrected data
    - base on RegisteredParsingObject to allow user-selection of projector
    - replace some code with calls to methods of DataSymmetriesForBins_PET_CartesianGrid

  * KT 25/11/2003 :
    - handle cases with less symmetries, e.g. when num_views%4!=0
    - Proj_Siddon has now a slightly different calling interface
      (templating should speed it up a tiny bit)
*/
/* this file still needs some cleaning. Sorry.
   and more DOC of course

   Most of the ugly stuff is because Proj_Siddon is a translation of M Egger's code, and
   hence does not know about RelatedViewgrams etc. So, there is a step that
   gets data from a 4D Array and sticks it into the RelatedViewgrams.
   Ugly though.
*/

#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
// KT 20/06/2001 should now work for non-arccorrected data as well
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange4D.h"
#include "stir/Array.h"
#include "stir/round.h"

#include <algorithm>
using std::min;
using std::max;

START_NAMESPACE_STIR

const char * const 
ForwardProjectorByBinUsingRayTracing::registered_name =
  "Ray Tracing";


void
ForwardProjectorByBinUsingRayTracing::
set_defaults()
{
  restrict_to_cylindrical_FOV = true;
}

void
ForwardProjectorByBinUsingRayTracing::
initialise_keymap()
{
  parser.add_start_key("Forward Projector Using Ray Tracing Parameters");
  parser.add_key("restrict to cylindrical FOV", &restrict_to_cylindrical_FOV);
  parser.add_stop_key("End Forward Projector Using Ray Tracing Parameters");
}

ForwardProjectorByBinUsingRayTracing::
  ForwardProjectorByBinUsingRayTracing()
{
  set_defaults();
}

ForwardProjectorByBinUsingRayTracing::
  ForwardProjectorByBinUsingRayTracing(
				   const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
                                   const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
{
  set_defaults();
  set_up(proj_data_info_ptr, image_info_ptr);
}

void
ForwardProjectorByBinUsingRayTracing::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
{
  ForwardProjectorByBin::set_up(proj_data_info_ptr, image_info_ptr);
  
  if (proj_data_info_ptr->get_num_views()%2 != 0)
    {
      error("The on-the-fly Ray tracing forward projector cannot handle data with odd number of views. Use another projector. Sorry.");
    }

  symmetries_ptr.reset(new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr, image_info_ptr));

  // check if data are according to what we can handle

  const VoxelsOnCartesianGrid<float> * vox_image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (image_info_ptr.get());

  if (vox_image_info_ptr == NULL)
    error("ForwardProjectorByBinUsingRayTracing initialised with a wrong type of DiscretisedDensity\n");

  const CartesianCoordinate3D<float> voxel_size = vox_image_info_ptr->get_voxel_size();

  const float sampling_distance_of_adjacent_LORs_xy =
    proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0));


  // z_origin_in_planes should be an integer
  const float z_origin_in_planes =
    image_info_ptr->get_origin().z()/voxel_size.z();
  if (fabs(round(z_origin_in_planes) - z_origin_in_planes) > 1.E-4)
    error("ForwardProjectorByBinUsingRayTracing: the shift in the "
          "z-direction of the origin (which is %g) should be a multiple of the plane "
          "separation (%g)\n",
          image_info_ptr->get_origin().z(), voxel_size.z());

  // num_planes_per_axial_pos should currently be an integer
  for (int segment_num = proj_data_info_ptr->get_min_segment_num();
       segment_num <= proj_data_info_ptr->get_max_segment_num();
       ++segment_num)
  {
		const float num_planes_per_axial_pos =
		  symmetries_ptr->get_num_planes_per_axial_pos(segment_num);
		if (fabs(round(num_planes_per_axial_pos) - num_planes_per_axial_pos) > 1.E-4)
		  error("ForwardProjectorByBinUsingRayTracing: the number of image planes "
				"per axial_pos (which is %g for segment %d) should be an integer\n",
				 num_planes_per_axial_pos, segment_num);
  }
  

  // KT 20/06/2001 converted from assert to a warning
  if(sampling_distance_of_adjacent_LORs_xy > voxel_size.x() + 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy > voxel_size.y() + 1.E-3)
     warning("ForwardProjectorByBinUsingRayTracing assumes that pixel size (in x,y) "
             "is greater than or equal to the bin size.\n"
             "As this is NOT the case with the current data, the projector will "
             "completely miss some voxels for some (or all) views.");
}


const DataSymmetriesForViewSegmentNumbers * 
ForwardProjectorByBinUsingRayTracing::get_symmetries_used() const
{
  if (!this->_already_set_up)
    error("ForwardProjectorByBin method called without calling set_up first.");
  return symmetries_ptr.get(); 
}

void 
ForwardProjectorByBinUsingRayTracing::
actual_forward_project(RelatedViewgrams<float>& viewgrams, 
		     const DiscretisedDensity<3,float>& density,
		     const int min_axial_pos_num, const int max_axial_pos_num,
		     const int min_tangential_pos_num, const int max_tangential_pos_num)

{
  // this will throw an exception when the cast does not work
  const VoxelsOnCartesianGrid<float>& image = 
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(density);

  const int num_views = viewgrams.get_proj_data_info_ptr()->get_num_views();

  if (viewgrams.get_basic_segment_num() == 0)
  {    
    if (viewgrams.get_num_viewgrams() == 1)
    {
      Viewgram<float> & pos_view = *viewgrams.begin();
      forward_project_view_2D(
	pos_view,  
	image,
	min_axial_pos_num, max_axial_pos_num,
	min_tangential_pos_num, max_tangential_pos_num);

    }
    else
    if (viewgrams.get_num_viewgrams() == 2)
    {
      assert(viewgrams.get_basic_view_num() >= 0);
      assert(viewgrams.get_basic_view_num() < num_views/2);
      Viewgram<float> & pos_view = *viewgrams.begin();
      if ((viewgrams.begin()+1)->get_view_num() == pos_view.get_view_num() + num_views/2)
	{
	  Viewgram<float> & pos_plus90 =*(viewgrams.begin()+1);
	  if (pos_plus90.get_view_num() != pos_view.get_view_num() + num_views/2)
	    error("ForwardProjectorUsingRayTracing: error in symmetries. Check 2D case with 2 viewgrams\n");

	  forward_project_view_plus_90_2D(
						    pos_view, pos_plus90,  
						    image,
						    min_axial_pos_num, max_axial_pos_num,
						    min_tangential_pos_num, max_tangential_pos_num);
	}
      else
	{
	  Viewgram<float> & pos_min180 =*(viewgrams.begin()+1);
	  if (pos_min180.get_view_num() != num_views - pos_view.get_view_num())
	    error("ForwardProjectorUsingRayTracing: error in symmetries. Check 2D case with 2 viewgrams\n");

	  forward_project_view_min_180_2D(
						    pos_view, pos_min180,  
						    image,
						    min_axial_pos_num, max_axial_pos_num,
						    min_tangential_pos_num, max_tangential_pos_num);
	}	  
    }
    else
    {
      assert(viewgrams.get_basic_view_num() > 0);
      assert(viewgrams.get_basic_view_num() < num_views/4);
      Viewgram<float> & pos_view = *(viewgrams.begin());
      Viewgram<float> & pos_plus90 =*(viewgrams.begin()+1);
      Viewgram<float> & pos_min180 =*(viewgrams.begin()+2);
      Viewgram<float> & pos_min90 =*(viewgrams.begin()+3);

      if (pos_plus90.get_view_num() != pos_view.get_view_num() + num_views/2)
	error("ForwardProjectorUsingRayTracing: error in symmetries. Check 2D case with 4 viewgrams\n");
      if (pos_min180.get_view_num() != num_views - pos_view.get_view_num())
	error("ForwardProjectorUsingRayTracing: error in symmetries. Check 2D case with 4 viewgrams\n");
      if (pos_min90.get_view_num() != num_views/2 - pos_view.get_view_num())
	error("ForwardProjectorUsingRayTracing: error in symmetries. Check 2D case with 4 viewgrams\n");

      forward_project_all_symmetries_2D(
	pos_view, pos_plus90, 
	pos_min180, pos_min90, 
	image,
	min_axial_pos_num, max_axial_pos_num,
	min_tangential_pos_num, max_tangential_pos_num);

    }
  }
  else
  {
    // segment symmetry
    if (viewgrams.get_num_viewgrams() == 2)
      {
	Viewgram<float> & pos_view = *(viewgrams.begin()+0);
	Viewgram<float> & neg_view =*(viewgrams.begin()+1);

	if (pos_view.get_view_num() != neg_view.get_view_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 2 viewgrams\n");
	if (neg_view.get_segment_num() != - pos_view.get_segment_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 2 viewgrams\n");

	forward_project_delta(
			      pos_view, neg_view,
			      image,
			      min_axial_pos_num, max_axial_pos_num,
			      min_tangential_pos_num, max_tangential_pos_num);
      }
    else if (viewgrams.get_num_viewgrams() == 4)
      {
	Viewgram<float> & pos_view = *(viewgrams.begin()+0);
	Viewgram<float> & neg_view =*(viewgrams.begin()+1);
	if (neg_view.get_view_num() != pos_view.get_view_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 2 viewgrams\n");
	if (neg_view.get_segment_num() != - pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");

	if ((viewgrams.begin()+2)->get_view_num() == pos_view.get_view_num() + num_views/2)
	  {
	    Viewgram<float> & pos_plus90 =*(viewgrams.begin()+2);
	    Viewgram<float> & neg_plus90 =*(viewgrams.begin()+3);
	    if (pos_plus90.get_view_num() != pos_view.get_view_num() + num_views/2)
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");
	    if (neg_plus90.get_view_num() != neg_view.get_view_num() + num_views/2)
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");
	    if (pos_plus90.get_segment_num() != pos_view.get_segment_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");
	    if (neg_plus90.get_segment_num() != - pos_view.get_segment_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");
	    forward_project_view_plus_90_and_delta(
						   pos_view, neg_view, pos_plus90, neg_plus90, 
						   image,
						   min_axial_pos_num, max_axial_pos_num,
						   min_tangential_pos_num, max_tangential_pos_num);
	  }
	else
	  {
	    Viewgram<float> & pos_min180 =*(viewgrams.begin()+2);
	    Viewgram<float> & neg_min180 =*(viewgrams.begin()+3);
	    if (pos_min180.get_view_num() != num_views - pos_view.get_view_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");	   
	    if (neg_min180.get_view_num() != num_views - neg_view.get_view_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");
	    if (pos_min180.get_segment_num() != pos_view.get_segment_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");
	    if (neg_min180.get_segment_num() != - pos_view.get_segment_num())
	      error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 4 viewgrams\n");

	    forward_project_view_min_180_and_delta(
						   pos_view, neg_view, pos_min180, neg_min180, 
						   image,
						   min_axial_pos_num, max_axial_pos_num,
						   min_tangential_pos_num, max_tangential_pos_num);	   
	  }
      }
    else if (viewgrams.get_num_viewgrams() == 8)
      {
	assert(viewgrams.get_basic_view_num() > 0);
	assert(viewgrams.get_basic_view_num() < num_views/4);
	Viewgram<float> & pos_view = *(viewgrams.begin()+0);
	Viewgram<float> & neg_view =*(viewgrams.begin()+1);
	Viewgram<float> & pos_plus90 =*(viewgrams.begin()+2);
	Viewgram<float> & neg_plus90 =*(viewgrams.begin()+3);
	Viewgram<float> & pos_min180 =*(viewgrams.begin()+4);
	Viewgram<float> & neg_min180=*(viewgrams.begin()+5);
	Viewgram<float> & pos_min90=*(viewgrams.begin()+6);
	Viewgram<float> & neg_min90 =*(viewgrams.begin()+7);

	if (pos_plus90.get_view_num() != pos_view.get_view_num() + num_views/2)
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (pos_min180.get_view_num() != num_views - pos_view.get_view_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (pos_min90.get_view_num() != num_views/2 - pos_view.get_view_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");

	if (neg_view.get_view_num() != pos_view.get_view_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (neg_min90.get_view_num() != num_views/2 - neg_view.get_view_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (neg_plus90.get_view_num() != neg_view.get_view_num() + num_views/2)
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (neg_min180.get_view_num() != num_views - neg_view.get_view_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");

	if (pos_plus90.get_segment_num() != pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (pos_min90.get_segment_num() != pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (pos_min180.get_segment_num() != pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");

	if (neg_view.get_segment_num() != - pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (neg_plus90.get_segment_num() != - pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (neg_min90.get_segment_num() != - pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");
	if (neg_min180.get_segment_num() != - pos_view.get_segment_num())
	  error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case with 8 viewgrams\n");

	forward_project_all_symmetries(
				       pos_view, neg_view, pos_plus90, neg_plus90, 
				       pos_min180, neg_min180, pos_min90, neg_min90,
				       image,
				       min_axial_pos_num, max_axial_pos_num,
				       min_tangential_pos_num, max_tangential_pos_num);

      }
    else // other number of viewgrams
      {
	error("ForwardProjectorUsingRayTracing: error in symmetries. Check 3D case\n");
      }

  } // oblique case


}



/*
    The version which uses all possible symmetries.
    Here 0<=view < num_views/4 (= 45 degrees)
*/

void 
ForwardProjectorByBinUsingRayTracing::
forward_project_all_symmetries(
			       Viewgram<float> & pos_view, 
			       Viewgram<float> & neg_view, 
			       Viewgram<float> & pos_plus90, 
			       Viewgram<float> & neg_plus90, 
			       Viewgram<float> & pos_min180, 
			       Viewgram<float> & neg_min180, 
			       Viewgram<float> & pos_min90, 
			       Viewgram<float> & neg_min90, 
			       const VoxelsOnCartesianGrid<float>& image,
			       const int min_ax_pos_num, const int max_ax_pos_num,
			       const int min_tangential_pos_num, const int max_tangential_pos_num) const
{

  // KT 20/06/2001 should now work for non-arccorrected data as well
  const ProjDataInfoCylindrical * proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindrical *>
    (pos_view.get_proj_data_info_ptr());
  if (proj_data_info_ptr == NULL)
    error("ForwardProjectorByBinUsingRayTracing::forward_project called with wrong type of ProjDataInfo\n");
    
  const int nviews = pos_view.get_proj_data_info_ptr()->get_num_views(); 
  
  const int segment_num = pos_view.get_segment_num();
  const int timing_pos_num = pos_view.get_timing_pos_num();
  const float delta = proj_data_info_ptr->get_average_ring_difference(segment_num);  
  const int view = pos_view.get_view_num();

  assert(delta > 0);

  assert(view >= 0);
  /* removed assertions which would break the temporary 2,4 parameter forward_project 
  assert(view <= view90);
  
  assert(pos_plus90.get_view_num() == nviews / 2 + pos_view.get_view_num());
  assert(pos_min90.get_view_num() == nviews / 2 - pos_view.get_view_num());
  assert(pos_min180.get_view_num() == nviews - pos_view.get_view_num());
  
  assert(neg_view.get_view_num() == pos_view.get_view_num());
  assert(neg_plus90.get_view_num() == pos_plus90.get_view_num());
  assert(neg_min90.get_view_num() == pos_min90.get_view_num());
  assert(neg_min180.get_view_num() == pos_min180.get_view_num());
  */
   //assert(image.get_min_z() == 0);

  assert(delta ==
    -proj_data_info_ptr->get_average_ring_difference(neg_view.get_segment_num()));
  
  // KT 21/05/98 added const where possible
  // TODO C value depends whether you are in Double or not,
  // If double C==2 => do 2*ax_pos0 and 2*ax_pos0+1
  const int C=1;
  
  int  D, tang_pos_num;
  int ax_pos0, my_ax_pos0;
  const float R = proj_data_info_ptr->get_ring_radius();
  
  // a variable which will be used in the loops over tang_pos_num to get s_in_mm
  Bin bin(pos_view.get_segment_num(), pos_view.get_view_num(),min_ax_pos_num,0,pos_view.get_timing_pos_num());    

  // KT 20/06/2001 rewrote using get_phi  
  const float cphi = cos(proj_data_info_ptr->get_phi(bin));
  const float sphi = sin(proj_data_info_ptr->get_phi(bin));

  // KT 20/06/2001 write using symmetries member
  // find correspondence between ring coordinates and image coordinates:
  // z = num_planes_per_axial_pos * ring + axial_pos_to_z_offset
  const int num_planes_per_axial_pos =
    round(symmetries_ptr->get_num_planes_per_axial_pos(segment_num));
  const float axial_pos_to_z_offset = 
    symmetries_ptr->get_axial_pos_to_z_offset(segment_num);
      
  // KT 20/06/2001 parameters to find 'basic' range of tang_pos_num
  const int max_abs_tangential_pos_num = 
    max(max_tangential_pos_num, -min_tangential_pos_num);
  const int min_abs_tangential_pos_num = 
    max_tangential_pos_num<0 ?
    -max_tangential_pos_num
    : (min_tangential_pos_num>0 ?
       min_tangential_pos_num
       : 0 );
  // in the loop, the case tang_pos_num==0 will be treated separately (because it's self-symmetric)
  const int min_tang_pos_num_in_loop =
    min_abs_tangential_pos_num==0 ? 1 : min_abs_tangential_pos_num;
  
  start_timers();

 
    Array <4,float> Projall(IndexRange4D(min_ax_pos_num, max_ax_pos_num, 0, 1, 0, 1, 0, 3));
    // KT 21/05/98 removed as now automatically zero 
    // Projall.fill(0);

    // In the case that axial sampling for the projection data = 2*voxel_size.z()
    // we draw 2 LORs, at -1/4 and +1/4 of the centre of the bin
    // If we don't do this, we will miss voxels in the forward projection step.

    // When the axial sampling is the same as the voxel size, we take just
    // 1 LOR.    
    float offset_start = -.25F;
    float offset_incr = .5F;

    int num_lors_per_virtual_ring = 2;
    
    if (num_planes_per_axial_pos == 1)
    {
        offset_start = 0;
        offset_incr=1;
	num_lors_per_virtual_ring = 1;
    }


    for (float offset = offset_start; offset < 0.3; offset += offset_incr)
    {
        if (view == 0 || 4*view == nviews ) {	/* phi=0 or 45 */
            for (D = 0; D < C; D++) {
	      if (min_abs_tangential_pos_num==0)
		{
		  /* Here tang_pos_num=0 and phi=0 or 45*/

		  if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		      < 2>(
#else
			   (2,
#endif
			   Projall, image, proj_data_info_ptr, cphi, sphi,
				      delta + D, 0, R,min_ax_pos_num, max_ax_pos_num,
				      offset, num_planes_per_axial_pos, axial_pos_to_z_offset,
				      1.F / num_lors_per_virtual_ring,
				      restrict_to_cylindrical_FOV))
		  for (ax_pos0 = min_ax_pos_num; ax_pos0 <= max_ax_pos_num; ax_pos0++) {
                    my_ax_pos0 = C * ax_pos0 + D;
		    
                    pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]; 
                    pos_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][2]; 
                    neg_view[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][0]; 
                    neg_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][2]; 
		  }
		}
                    /* Now tang_pos_num!=0 and phi=0 or 45 */
                for (tang_pos_num = min_tang_pos_num_in_loop; tang_pos_num <= max_abs_tangential_pos_num; tang_pos_num++) 
		  {
		    bin.tangential_pos_num() = tang_pos_num;
		    const float s_in_mm = proj_data_info_ptr->get_s(bin);
                    if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
			<1>(
#else
			    (1,
#endif
			    Projall, image, proj_data_info_ptr, cphi, sphi,
				       delta + D, s_in_mm, R,min_ax_pos_num, max_ax_pos_num,
				       offset, num_planes_per_axial_pos, axial_pos_to_z_offset,
				       1.F/num_lors_per_virtual_ring,
				       restrict_to_cylindrical_FOV))
		      for (ax_pos0 = min_ax_pos_num; ax_pos0 <= max_ax_pos_num; ax_pos0++) {
                        my_ax_pos0 = C * ax_pos0 + D;
                        if (tang_pos_num<=max_tangential_pos_num)
			  {
			    pos_view[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][0]; 
			    pos_plus90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][2]; 
			    neg_view[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][1][0][0]; 
			    neg_plus90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][1][0][2]; 
			  }
			if (-tang_pos_num>=min_tangential_pos_num)
			  {
			    pos_view[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][0]; 
			    pos_plus90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][2]; 
			    neg_view[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][1][1][0]; 
			    neg_plus90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][1][1][2]; 
			  }
                    }
                }
            }
        } else {

         
            for (D = 0; D < C; D++) {
	      if (min_abs_tangential_pos_num==0)
		{             
		  /* Here tang_pos_num==0 and phi!=k*45 */
		  if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		      <4>(
#else
			  (4,
#endif
			  Projall, image, proj_data_info_ptr, cphi, sphi, 
				     delta + D, 0, R,min_ax_pos_num, max_ax_pos_num,
				     offset, num_planes_per_axial_pos, axial_pos_to_z_offset ,
				     1.F/num_lors_per_virtual_ring,
				     restrict_to_cylindrical_FOV))
		    for (ax_pos0 = min_ax_pos_num; ax_pos0 <= max_ax_pos_num; ax_pos0++) {
                    my_ax_pos0 = C * ax_pos0 + D;
                    pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]; 
                    pos_min90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][1]; 
                    pos_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][2]; 
                    pos_min180[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][3]; 
                    neg_view[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][0]; 
                    neg_min90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][1]; 
                    neg_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][2]; 
                    neg_min180[my_ax_pos0][0] +=  Projall[ax_pos0][1][0][3]; 
		  }
		}
		

                    /* Here tang_pos_num!=0 and phi!=k*45. */
	      for (tang_pos_num = min_tang_pos_num_in_loop; tang_pos_num <= max_abs_tangential_pos_num; tang_pos_num++) {
		    bin.tangential_pos_num() = tang_pos_num;
		    const float s_in_mm = proj_data_info_ptr->get_s(bin);

                    if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
			<3>(
#else
			    (3,
#endif
			    Projall, image, proj_data_info_ptr, cphi, sphi,
				       delta + D, s_in_mm, R,min_ax_pos_num, max_ax_pos_num,
				       offset, num_planes_per_axial_pos, axial_pos_to_z_offset ,
				       1.F/num_lors_per_virtual_ring,
				       restrict_to_cylindrical_FOV))
		      for (ax_pos0 = min_ax_pos_num; ax_pos0 <= max_ax_pos_num; ax_pos0++) 
		      {
			my_ax_pos0 = C * ax_pos0 + D;
			if (tang_pos_num<=max_tangential_pos_num)
			  {
			    pos_view[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][0]; 
			    pos_min90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][1]; 
			    pos_plus90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][2]; 
			    pos_min180[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][3]; 
			    neg_view[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][1][0][0]; 
			    neg_min90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][1][0][1]; 
			    neg_plus90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][1][0][2]; 
			    neg_min180[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][1][0][3]; 
			  }
			if (-tang_pos_num>=min_tangential_pos_num)
			  {
			    pos_view[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][0]; 
			    pos_min90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][1]; 
			    pos_plus90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][2]; 
			    pos_min180[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][3]; 
			    neg_view[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][1][1][0]; 
			    neg_min90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][1][1][1]; 
			    neg_plus90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][1][1][2]; 
			    neg_min180[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][1][1][3]; 
			  }
		      }   
                }     
            }


        }// end of } else {
    }// end of test for offset loop
  
    stop_timers();
  
}


/*
This function projects 2 viewgrams related by segment symmetry.
*/

void 
ForwardProjectorByBinUsingRayTracing::
  forward_project_delta(Viewgram<float> & pos_view, 
			Viewgram<float> & neg_view,  
			const VoxelsOnCartesianGrid<float> & image,
			const int min_axial_pos_num, const int max_axial_pos_num,
			const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  assert(pos_view.get_segment_num() > 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views());

  Viewgram<float> dummy = pos_view;


  forward_project_all_symmetries(
				 pos_view, 
				 neg_view, 
				 dummy,
				 dummy,
				 dummy,
				 dummy,
				 dummy,
				 dummy,
				 image,
				 min_axial_pos_num, max_axial_pos_num,
				 min_tangential_pos_num, max_tangential_pos_num);
}


/*
This function projects 4 viewgrams related by symmetry.
  Here 0<=view < num_views/2 (= 90 degrees)
*/

void 
ForwardProjectorByBinUsingRayTracing::
  forward_project_view_plus_90_and_delta(Viewgram<float> & pos_view, 
				         Viewgram<float> & neg_view, 
				         Viewgram<float> & pos_plus90, 
				         Viewgram<float> & neg_plus90, 
				         const VoxelsOnCartesianGrid<float> & image,
				         const int min_axial_pos_num, const int max_axial_pos_num,
				         const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  assert(pos_view.get_segment_num() > 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/2);

  Viewgram<float> dummy = pos_view;


  forward_project_all_symmetries(
				 pos_view, 
				 neg_view, 
				 pos_plus90, 
				 neg_plus90, 
				 dummy,
				 dummy,
				 dummy,
				 dummy,
				 image,
				 min_axial_pos_num, max_axial_pos_num,
				 min_tangential_pos_num, max_tangential_pos_num);
}


void 
ForwardProjectorByBinUsingRayTracing::
  forward_project_view_min_180_and_delta(Viewgram<float> & pos_view, 
				         Viewgram<float> & neg_view, 
				         Viewgram<float> & pos_min180, 
				         Viewgram<float> & neg_min180, 
				         const VoxelsOnCartesianGrid<float> & image,
				         const int min_axial_pos_num, const int max_axial_pos_num,
				         const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  assert(pos_view.get_segment_num() > 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/2);

  Viewgram<float> dummy = pos_view;

  forward_project_all_symmetries(
				 pos_view, 
				 neg_view, 
				 dummy,
				 dummy,
				 pos_min180, 
				 neg_min180, 
				 dummy,
				 dummy,
				 image,
				 min_axial_pos_num, max_axial_pos_num,
				 min_tangential_pos_num, max_tangential_pos_num);
}

#if 0
// old specific 2D versions, worked plane by plane
// this still assumes odd ranges for tang_pos_num etc.

void ForwardProjectorByBinUsingRayTracing::forward_project_2D(Segment<float> &sinos,const VoxelsOnCartesianGrid<float>& image,
			const int view, const int rmin, const int rmax)
{

  int segment_num =sinos.get_segment_num();

  const ProjDataInfoCylindricalArcCorr* proj_data_cyl_ptr =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr*>(sinos.get_proj_data_info_ptr());

  if ( proj_data_cyl_ptr==NULL)
  {
    error("ForwardProjectorByBinUsingRayTracing::Casting failed\n");
  }

  assert(proj_data_cyl_ptr->get_average_ring_difference(segment_num) ==0);

  // CL&KT 05/11/98 use scan_info
  const float planes_per_virtual_ring = 
    sinos.get_proj_data_info_ptr()->get_scanner_ptr()->ring_spacing / image.get_voxel_size().z();
    
  int num_planes_per_axial_pos = (int)(planes_per_virtual_ring + 0.5);
  assert(planes_per_virtual_ring > 0);
  // Check if planes_per_ring is very close to an int
  assert(fabs(num_planes_per_axial_pos / planes_per_virtual_ring - 1) < 1e-5);

   // check if there is axial compression  
  if (proj_data_cyl_ptr->get_max_ring_difference(segment_num) != 0)
    {
      //TODO handle axial compression in a different way
      num_planes_per_axial_pos = 1;
    }

  // We support only 1 or 2 planes per ring now
  assert(num_planes_per_axial_pos == 1 || num_planes_per_axial_pos == 2);

  // Initialise a 2D sinogram here to avoid reallocating it every ring
  // We use get_sinogram to get correct sizes etc., but 
  // data will be overwritten
  Sinogram<float> sino = sinos.get_sinogram(sinos.get_min_axial_pos());

  // First do direct planes
  {
    for (int ax_pos = rmin; ax_pos <= rmax; ax_pos++)
      {	
	sino = sinos.get_sinogram(ax_pos);
	forward_project_2D(sino,image, num_planes_per_axial_pos*ax_pos, view);
	sinos.set_sinogram(sino);
      }
  }

  // Now do indirect planes
  if (num_planes_per_axial_pos == 2)
    {
 
      // TODO terribly inefficient as this is repeated for every view
      // adding in lots of zeroes
      for (int  ax_pos = rmin; ax_pos < rmax; ax_pos++)
	{	
	  sino.fill(0);
	  // forward project the indirect plane
	  forward_project_2D(sino,image, num_planes_per_axial_pos*ax_pos+1, view);
	  
	  // add 'half' of the sinogram to the 2 neighbouring rings
	  sino /= 2;

	  Sinogram<float> sino_tmp = sinos.get_sinogram(ax_pos);
	  sino_tmp += sino;
	  sinos.set_sinogram(sino_tmp);

	  sino_tmp = sinos.get_sinogram(ax_pos+1);
	  sino_tmp += sino;
	  sinos.set_sinogram(sino_tmp);
	}
    }
}

void ForwardProjectorByBinUsingRayTracing::forward_project_2D(Sinogram<float> &sino,const VoxelsOnCartesianGrid<float>&image, 
							      const int plane_num, const int view)
{
  const ProjDataInfoCylindricalArcCorr* proj_data_cyl_ptr =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr*>(sino.get_proj_data_info_ptr());
  
  if ( proj_data_cyl_ptr==NULL)
  {
    error("ForwardProjectorByBinUsingRayTracing::Casting failed");
    
    int segmnet_num = sino.get_segment_num();
    
    // KT 06/10/98 added num_planes_per_axial_pos stuff for normalisation later on
    assert(proj_data_cyl_ptr->get_average_ring_difference(segmnet_num) ==0);
    
    // KT&CL 21/12/99 changed name from planes_per_ring to planes_per_virtual_ring
    const float planes_per_virtual_ring = 
      sino.get_proj_data_info_ptr()->get_scanner_ptr()->ring_spacing / image.get_voxel_size().z();
    
    int num_planes_per_axial_pos = (int)(planes_per_virtual_ring + 0.5);
    assert(planes_per_virtual_ring > 0);
    // Check if planes_per_ring is very close to an int
    assert(fabs(num_planes_per_axial_pos / planes_per_virtual_ring - 1) < 1e-5);
    
    // check if there is axial compression  
    if (proj_data_cyl_ptr->get_max_ring_difference(segmnet_num) != 0)
    {
      //TODO find out about axial compression in a different way
      num_planes_per_axial_pos = 1;
    }
    
    // We support only 1 or 2 planes per ring now
    assert(num_planes_per_axial_pos == 1 || num_planes_per_axial_pos == 2);
    
    // CL&KT 05/11/98 use scan_info
    const int nviews = sino.get_proj_data_info_ptr()->get_num_views();
    const int view90 = nviews / 2;
    const int view45 = view90 / 2;
    const int plus90 = view90+view;
    const int min180 = nviews-view;
    const int min90 = view90-view;
    
    
    assert(sino.get_num_tangential_poss()  == image.get_x_size());
    assert(image.get_min_x() == -image.get_max_x());
    // CL&KT 05/11/98 use scan_info, enable assert
    assert(image.get_voxel_size().x == sino.get_proj_data_info_ptr()->get_num_tangential_poss()); 
    assert(sino.get_max_tangential_pos_num() == -sino.get_min_tangential_pos_num());
    
    
    // TODO C value depends whether you are in Double or not,
    // If double C==2 => do 2*ax_pos0 and 2*ax_pos0+1
    const int C=1;
    
    int  D, tang_pos_num;
    // CL&KT 05/11/98 use scan_info
    const float R = sino.get_proj_data_info_ptr()->get_scanner_ptr()->ring_radius;
    

    const float itophi = _PI / nviews;
    
    const float cphi = cos(view * itophi);
    const float sphi = sin(view * itophi);
    
    const int   projrad = (int) (sino.get_num_tangential_poss() / 2) - 1;
    
    start_timers();
    
    //TODO for the moment, just handle 1 plane and use some 3D variables 
    const int min_ax_pos = 0;
    const int max_ax_pos = 0;
    const float delta = 0;
    int ax_pos0;
    int my_ax_pos0;
    
    Array <4,float> Projall(min_ax_pos, max_ax_pos, 0, 1, 0, 1, 0, 3);
    
    // only 1 value of offset for 2D case.
    // However, we have to divide projections by num_planes_per_axial_pos
    // to get values equal to the (average) line integral
    
    // KT&CL 21/12/99 use num_planes_per_axial_pos in the offset and in if proj_Siddon
    // inside proj_Siddon z=num_planes_per_axial_pos*offset, but that has 
    // to be plane_num, so we set offset accordingly
    const float offset = float(plane_num)/num_planes_per_axial_pos;
    {
      
      if (view == 0 || view == view45 ) {	/* phi=0 or 45 */
	for (D = 0; D < C; D++) {
	  /* Here tang_pos_num=0 and phi=0 or 45*/
	  
	  if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
	      <2>(
#else
		  (2,
#endif
		 Projall,image,proj_data_cyl_ptr, 
			      cphi, sphi, delta + D, 0, R,min_ax_pos, max_ax_pos, 
			      offset, num_planes_per_axial_pos, 0,
			      1.F/num_lors_per_virtual_ring,
			      restrict_to_cylindrical_FOV))
	    for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
	      my_ax_pos0 = C * ax_pos0 + D;
	      sino[view][0] += Projall[ax_pos0][0][0][0] / num_planes_per_axial_pos; 
	      sino[plus90][0] += Projall[ax_pos0][0][0][2] / num_planes_per_axial_pos;
	    }
	  /* Now tang_pos_num!=0 and phi=0 or 45 */
	  for (tang_pos_num = min_tang_pos_num_in_loop; tang_pos_num <= max_abs_tangential_pos_num; tang_pos_num++) {
	    if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		<1>(
#else
		    (1,
#endif
		    Projall,image,proj_data_cyl_ptr, cphi, sphi,
			       delta + D, tang_pos_num, R,min_ax_pos,max_ax_pos, 
			       offset, num_planes_per_axial_pos, 0,
			       1.F/num_lors_per_virtual_ring,
			       restrict_to_cylindrical_FOV))
	      for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
		ax_pos0 = C * ax_pos0 + D;
		sino[view][tang_pos_num] += Projall[ax_pos0][0][0][0] / num_planes_per_axial_pos;
		sino[plus90][tang_pos_num] += Projall[ax_pos0][0][0][2] / num_planes_per_axial_pos;
		sino[view][-tang_pos_num] += Projall[ax_pos0][0][1][0] / num_planes_per_axial_pos;
		sino[plus90][-tang_pos_num] += Projall[ax_pos0][0][1][2] / num_planes_per_axial_pos;
	      }
	  }
	}
      } else {
	
	
	for (D = 0; D < C; D++) {
	  /* Here tang_pos_num==0 and phi!=k*45 */
	  if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
	      <4>(
#else
		  (4,
#endif
		  Projall,image,proj_data_cyl_ptr, cphi, sphi,
			     delta + D, 0, R,min_ax_pos,max_ax_pos, 
			     offset, num_planes_per_axial_pos, 0,
			     1.F/num_lors_per_virtual_ring,
			     restrict_to_cylindrical_FOV))
	    for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
	    my_ax_pos0 = C * ax_pos0 + D;
	    sino[view][0] += Projall[ax_pos0][0][0][0] / num_planes_per_axial_pos;
	    sino[min90][0] += Projall[ax_pos0][0][0][1] / num_planes_per_axial_pos;
	    sino[plus90][0] += Projall[ax_pos0][0][0][2] / num_planes_per_axial_pos;
	    sino[min180][0] += Projall[ax_pos0][0][0][3] / num_planes_per_axial_pos;
	  }
	  
	  /* Here tang_pos_num!=0 and phi!=k*45. */
	  for (tang_pos_num = min_tang_pos_num_in_loop; tang_pos_num <= max_abs_tangential_pos_num; tang_pos_num++) {
	    if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		<3>(
#else
		    (3,
#endif
		    Projall,image,proj_data_cyl_ptr, cphi, sphi,
			       delta + D, tang_pos_num, R,min_ax_pos, max_ax_pos, 
			       offset, num_planes_per_axial_pos, 0,
			       1.F/num_lors_per_virtual_ring,
			       restrict_to_cylindrical_FOV))
	      for (ax_pos0 = min_ax_pos; ax_pos0 <= max_ax_pos; ax_pos0++) {
	      my_ax_pos0 = C * ax_pos0 + D;
	      sino[view][tang_pos_num] += Projall[ax_pos0][0][0][0] / num_planes_per_axial_pos;
	      sino[min90][tang_pos_num] += Projall[ax_pos0][0][0][1] / num_planes_per_axial_pos;
	      sino[plus90][tang_pos_num] += Projall[ax_pos0][0][0][2] / num_planes_per_axial_pos;
	      sino[min180][tang_pos_num] += Projall[ax_pos0][0][0][3] / num_planes_per_axial_pos;
	      sino[view][-tang_pos_num] += Projall[ax_pos0][0][1][0] / num_planes_per_axial_pos;
	      sino[min90][-tang_pos_num] += Projall[ax_pos0][0][1][1] / num_planes_per_axial_pos;
	      sino[plus90][-tang_pos_num] += Projall[ax_pos0][0][1][2] / num_planes_per_axial_pos;
	      sino[min180][-tang_pos_num] += Projall[ax_pos0][0][1][3] / num_planes_per_axial_pos;
	    }   
	  }     
	}
	
	
      }// end of } else {
    }// end of test for offset loop
    
    stop_timers();
    
}

#endif // old 2D versions

void 
ForwardProjectorByBinUsingRayTracing::
forward_project_view_2D(Viewgram<float> & pos_view, 
			const VoxelsOnCartesianGrid<float> & image,
			const int min_axial_pos_num, const int max_axial_pos_num,
			const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  assert(pos_view.get_segment_num() == 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views());

  Viewgram<float> dummy = pos_view;

  forward_project_all_symmetries_2D(
				    pos_view, 
				    dummy, 
				    dummy,
				    dummy,             
				    image,
				    min_axial_pos_num, max_axial_pos_num,
				    min_tangential_pos_num, max_tangential_pos_num);

}

void 
ForwardProjectorByBinUsingRayTracing::
forward_project_view_plus_90_2D(Viewgram<float> & pos_view, 
				Viewgram<float> & pos_plus90, 
				const VoxelsOnCartesianGrid<float> & image,
				const int min_axial_pos_num, const int max_axial_pos_num,
				const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  assert(pos_view.get_segment_num() == 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/2);

  Viewgram<float> dummy = pos_view;

  forward_project_all_symmetries_2D(
				    pos_view, 
				    pos_plus90, 
				    dummy,
				    dummy,             
				    image,
				    min_axial_pos_num, max_axial_pos_num,
				    min_tangential_pos_num, max_tangential_pos_num);
}


void 
ForwardProjectorByBinUsingRayTracing::
forward_project_view_min_180_2D(Viewgram<float> & pos_view, 
			       Viewgram<float> & pos_min180, 
			       const VoxelsOnCartesianGrid<float> & image,
			       const int min_axial_pos_num, const int max_axial_pos_num,
			       const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  assert(pos_view.get_segment_num() == 0);
  assert(pos_view.get_view_num() >= 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/2);

  Viewgram<float> dummy = pos_view;

  forward_project_all_symmetries_2D(
				    pos_view, 
				    dummy,
				    pos_min180, 
				    dummy,             
				    image,
				    min_axial_pos_num, max_axial_pos_num,
				    min_tangential_pos_num, max_tangential_pos_num);
}



void 
ForwardProjectorByBinUsingRayTracing::
forward_project_all_symmetries_2D(Viewgram<float> & pos_view, 
			         Viewgram<float> & pos_plus90, 
			         Viewgram<float> & pos_min180, 
			         Viewgram<float> & pos_min90, 
			         const VoxelsOnCartesianGrid<float>& image,
			         const int min_axial_pos_num, const int max_axial_pos_num,
			         const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  start_timers();

  // KT 20/06/2001 should now work for non-arccorrected data as well
  const ProjDataInfoCylindrical * proj_data_info_ptr =
    dynamic_cast<const ProjDataInfoCylindrical *>
    (pos_view.get_proj_data_info_ptr());
  if (proj_data_info_ptr == NULL)
    error("ForwardProjectorByBinUsingRayTracing::forward_project called with wrong type of ProjDataInfo\n");
    
  const int nviews = pos_view.get_proj_data_info_ptr()->get_num_views(); 
  
  const int segment_num = pos_view.get_segment_num();
  const int timing_pos_num = pos_view.get_timing_pos_num();
  const float delta = proj_data_info_ptr->get_average_ring_difference(segment_num);  
  const int view = pos_view.get_view_num();

  assert(delta == 0);
  assert(view >= 0);
  

  /* remove assertions which would break the temporary 1,2 parameter forward_project.
     Now checked before calling 
  assert(pos_plus90.get_view_num() == nviews / 2 + pos_view.get_view_num());
  assert(pos_min90.get_view_num() == nviews / 2 - pos_view.get_view_num());
  assert(pos_min180.get_view_num() == nviews - pos_view.get_view_num());
  */
  //assert(neg_view.get_view_num() == pos_view.get_view_num());
  //assert(neg_plus90.get_view_num() == pos_plus90.get_view_num());
  //assert(neg_min90.get_view_num() == pos_min90.get_view_num());
  //assert(neg_min180.get_view_num() == pos_min180.get_view_num());
    
  // KT 21/05/98 added const where possible
  // TODO C value depends whether you are in Double or not,
  // If double C==2 => do 2*ax_pos0 and 2*ax_pos0+1
  const int C=1;
  
  int  D, tang_pos_num;
  int my_ax_pos0;
  const float R = proj_data_info_ptr->get_ring_radius();

  // a variable which will be used in the loops over tang_pos_num to get s_in_mm
  Bin bin(pos_view.get_segment_num(), pos_view.get_view_num(),min_axial_pos_num,0,pos_view.get_timing_pos_num());    
  
  // KT 20/06/2001 rewrote using get_phi  
  const float cphi = cos(proj_data_info_ptr->get_phi(bin));
  const float sphi = sin(proj_data_info_ptr->get_phi(bin));

  // find correspondence between ring coordinates and image coordinates:
  // z = num_planes_per_axial_pos * ax_pos_num + axial_pos_to_z_offset
  // KT 20/06/2001 write using symmetries member
  const int num_planes_per_axial_pos =
    round(symmetries_ptr->get_num_planes_per_axial_pos(segment_num));
  const float axial_pos_to_z_offset = 
    symmetries_ptr->get_axial_pos_to_z_offset(segment_num);
  
  const int max_abs_tangential_pos_num = 
    max(max_tangential_pos_num, -min_tangential_pos_num);

  const int min_abs_tangential_pos_num = 
    max_tangential_pos_num<0 ?
    -max_tangential_pos_num
    : (min_tangential_pos_num>0 ?
       min_tangential_pos_num
       : 0 );
  const int min_tang_pos_num_in_loop =
    min_abs_tangential_pos_num==0 ? 1 : min_abs_tangential_pos_num;
    

  
  Array <4,float> Projall(IndexRange4D(min_axial_pos_num, max_axial_pos_num, 0, 1, 0, 1, 0, 3));
  Array <4,float> Projall2(IndexRange4D(min_axial_pos_num, max_axial_pos_num+1, 0, 1, 0, 1, 0, 3));
  
  // What to do when num_planes_per_axial_pos==2 ?
  // In the 2D case, the approach followed in 3D is ill-defined, as we would be 
  // forward projecting right along the edges of the voxels.
  // Instead, we take for the contribution to an axial_pos_num, 
  // 1/2 left_voxel + centre_voxel + 1/2 right_voxel
  
  int num_lors_per_virtual_ring = 2;
  
  if (num_planes_per_axial_pos == 1)
  {
    num_lors_per_virtual_ring = 1;
  }
  
  
  
  if (view == 0 || 4*view == nviews ) 
  {	/* phi=0 or 45 */
    for (D = 0; D < C; D++)       
    { 
      if (min_abs_tangential_pos_num==0)
	{
	  /* Here tang_pos_num=0 and phi=0 or 45*/     
	  {        
	    if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		<2>(
#else
		    (2,
#endif
		    Projall, image, proj_data_info_ptr, cphi, sphi,
			       delta + D, 0, R,min_axial_pos_num, max_axial_pos_num,
			       0.F/*==offset*/, num_planes_per_axial_pos, axial_pos_to_z_offset ,
			       1.F/num_lors_per_virtual_ring,
			       restrict_to_cylindrical_FOV))
	      for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
	      {
		my_ax_pos0 = C * ax_pos0 + D;
		
		pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]; 
		pos_plus90[my_ax_pos0][0] +=Projall[ax_pos0][0][0][2]; 
	      }
	  }
	  
	  if (num_planes_per_axial_pos == 2)
	    {	 	  
	      if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		  <2>(
#else
		      (2,
#endif
		      Projall2, image, proj_data_info_ptr, cphi, sphi,
				 delta + D, 0, R, min_axial_pos_num,  max_axial_pos_num+1,
				 -0.5F/*==offset*/, num_planes_per_axial_pos, axial_pos_to_z_offset ,
				 1.F/4,
				 restrict_to_cylindrical_FOV))
		for (int ax_pos0 =  min_axial_pos_num; ax_pos0 <=  max_axial_pos_num; ax_pos0++) 
		{
		  my_ax_pos0 = C * ax_pos0 + D;
		  pos_view[my_ax_pos0][0] += (Projall2[ax_pos0+1][0][0][0]+ Projall2[ax_pos0][0][0][0]); 
		  pos_plus90[my_ax_pos0][0] += (Projall2[ax_pos0+1][0][0][2]+ Projall2[ax_pos0][0][0][2]);
		}	      
	    }
	}
      
      /* Now tang_pos_num!=0 and phi=0 or 45 */
      for (tang_pos_num = min_tang_pos_num_in_loop; tang_pos_num <= max_abs_tangential_pos_num; tang_pos_num++) 
      {
	bin.tangential_pos_num() = tang_pos_num;
	const float s_in_mm = proj_data_info_ptr->get_s(bin);
	

        {                              
          if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
	      <1>(
#else
		  (1,
#endif
		  Projall, image, proj_data_info_ptr, cphi, sphi,
			     delta + D, s_in_mm, R,min_axial_pos_num, max_axial_pos_num,
			     0.F, num_planes_per_axial_pos, axial_pos_to_z_offset,
			     1.F/num_lors_per_virtual_ring,
			     restrict_to_cylindrical_FOV))
	    for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
	      {
            my_ax_pos0 = C * ax_pos0 + D;
	    if (tang_pos_num<=max_tangential_pos_num)
	      {
		pos_view[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][0]; 
		pos_plus90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][2]; 
	      }
	    if (-tang_pos_num>=min_tangential_pos_num)
	      {
		pos_view[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][0]; 
		pos_plus90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][2]; 
	      }
          }
        }
        if (num_planes_per_axial_pos == 2)
        {                            
          if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
	      <1>(
#else
		  (1,
#endif
		  Projall2, image, proj_data_info_ptr, cphi, sphi,
			     delta + D, s_in_mm, R,min_axial_pos_num, max_axial_pos_num+1,
			     -0.5F, num_planes_per_axial_pos, axial_pos_to_z_offset,
			     1.F/4,
			     restrict_to_cylindrical_FOV))
	    for (int ax_pos0 =min_axial_pos_num; ax_pos0 <=max_axial_pos_num; ax_pos0++) 
	      {
            my_ax_pos0 = C * ax_pos0 + D;
	    if (tang_pos_num<=max_tangential_pos_num)
	      {
		pos_view[my_ax_pos0][tang_pos_num] +=(Projall2[ax_pos0][0][0][0]+Projall2[ax_pos0+1][0][0][0]); 
		pos_plus90[my_ax_pos0][tang_pos_num] += (Projall2[ax_pos0][0][0][2]+Projall2[ax_pos0+1][0][0][2]); 
	      }
	    if (-tang_pos_num>=min_tangential_pos_num)
	      {
		pos_view[my_ax_pos0][-tang_pos_num] +=(Projall2[ax_pos0][0][1][0]+Projall2[ax_pos0+1][0][1][0]); 
		pos_plus90[my_ax_pos0][-tang_pos_num] +=(Projall2[ax_pos0][0][1][2]+Projall2[ax_pos0+1][0][1][2]);
	      }            
          }
        }
      } // Loop over tang_pos_num      
    } // Loop over D
  }
  else 
  {
    // general phi    
    for (D = 0; D < C; D++) 
    {
      if (min_abs_tangential_pos_num==0)
	{
	  /* Here tang_pos_num==0 and phi!=k*45 */
	  {
	    if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		<4>(
#else
		    (4,
#endif
		    Projall, image, proj_data_info_ptr, cphi, sphi, 
			       delta + D, 0, R,min_axial_pos_num, max_axial_pos_num,
			       0.F, num_planes_per_axial_pos, axial_pos_to_z_offset ,
			       1.F/num_lors_per_virtual_ring,
			       restrict_to_cylindrical_FOV))
	      for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
	      {
		my_ax_pos0 = C * ax_pos0 + D;
		pos_view[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][0]; 
		pos_min90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][1]; 
		pos_plus90[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][2]; 
		pos_min180[my_ax_pos0][0] +=  Projall[ax_pos0][0][0][3]; 
	      }
	  }
	  
	  if (num_planes_per_axial_pos == 2)        
	    {         
	      if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
		  <4>(
#else
		      (4,
#endif
		      Projall2, image, proj_data_info_ptr, cphi, sphi, 
				 delta + D, 0, R,min_axial_pos_num, max_axial_pos_num,
				 -0.5F, num_planes_per_axial_pos, axial_pos_to_z_offset ,
				 1.F/4,
				 restrict_to_cylindrical_FOV))
		for (int ax_pos0 = min_axial_pos_num; ax_pos0 <=max_axial_pos_num; ax_pos0++) 
		{
		  my_ax_pos0 = C * ax_pos0 + D;
		  pos_view[my_ax_pos0][0] +=  (Projall2[ax_pos0][0][0][0]+Projall2[ax_pos0+1][0][0][0]); 
		  pos_min90[my_ax_pos0][0] += (Projall2[ax_pos0][0][0][1]+Projall2[ax_pos0+1][0][0][1]); 
		  pos_plus90[my_ax_pos0][0] +=(Projall2[ax_pos0][0][0][2]+Projall2[ax_pos0+1][0][0][2]); 
		  pos_min180[my_ax_pos0][0] +=(Projall2[ax_pos0][0][0][3]+Projall2[ax_pos0+1][0][0][3]); 
		}
	    }
	}
      
      /* Here tang_pos_num!=0 and phi!=k*45. */
      for (tang_pos_num = min_tang_pos_num_in_loop; tang_pos_num <= max_abs_tangential_pos_num; tang_pos_num++)         
      {
	bin.tangential_pos_num() = tang_pos_num;
	const float s_in_mm = proj_data_info_ptr->get_s(bin);

        {          
          if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
	      <3>(
#else
		  (3,
#endif
		  Projall, image, proj_data_info_ptr, cphi, sphi,
			     delta + D, s_in_mm, R,min_axial_pos_num, max_axial_pos_num,
			     0.F, num_planes_per_axial_pos, axial_pos_to_z_offset ,
			     1.F/num_lors_per_virtual_ring,
			     restrict_to_cylindrical_FOV))
	    for (int ax_pos0 = min_axial_pos_num; ax_pos0<= max_axial_pos_num; ax_pos0++) 
	      {
            my_ax_pos0 = C * ax_pos0 + D;
	    if (tang_pos_num<=max_tangential_pos_num)
	      {
		pos_view[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][0]; 
		pos_min90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][1]; 
		pos_plus90[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][2]; 
		pos_min180[my_ax_pos0][tang_pos_num] +=  Projall[ax_pos0][0][0][3]; 
	      }
	    if (-tang_pos_num>=min_tangential_pos_num)
	      {
		pos_view[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][0]; 
		pos_min90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][1]; 
		pos_plus90[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][2]; 
		pos_min180[my_ax_pos0][-tang_pos_num] +=  Projall[ax_pos0][0][1][3]; 
	      }
          }   
        } 
        if (num_planes_per_axial_pos == 2)
        {
	  if (proj_Siddon
#ifndef STIR_SIDDON_NO_TEMPLATE
	      <3>(
#else
		  (3,
#endif
		  Projall2, image, proj_data_info_ptr, cphi, sphi,
			     delta + D, s_in_mm, R,min_axial_pos_num, max_axial_pos_num+1,
			     -0.5F, num_planes_per_axial_pos, axial_pos_to_z_offset ,
			     1.F/4,
			     restrict_to_cylindrical_FOV))
	    for (int ax_pos0 = min_axial_pos_num; ax_pos0 <= max_axial_pos_num; ax_pos0++) 
	    {
	      my_ax_pos0 = C * ax_pos0 + D;
	      if (tang_pos_num<=max_tangential_pos_num)
		{
		  pos_view[ my_ax_pos0][tang_pos_num] +=(Projall2[ax_pos0][0][0][0]+Projall2[ax_pos0+1][0][0][0]); 
		  pos_min90[my_ax_pos0][tang_pos_num] += (Projall2[ax_pos0][0][0][1]+Projall2[ax_pos0+1][0][0][1]); 
		  pos_plus90[ my_ax_pos0][tang_pos_num] +=(Projall2[ax_pos0][0][0][2]+Projall2[ax_pos0+1][0][0][2]); 
		  pos_min180[ my_ax_pos0][tang_pos_num] += (Projall2[ax_pos0][0][0][3]+Projall2[ax_pos0+1][0][0][3]); 
		}
	      if (-tang_pos_num>=min_tangential_pos_num)
		{
		  pos_view[ my_ax_pos0][-tang_pos_num] +=  (Projall2[ax_pos0][0][1][0] +Projall2[ax_pos0+1][0][1][0]); 
		  pos_min90[ my_ax_pos0][-tang_pos_num] +=(Projall2[ax_pos0][0][1][1]+Projall2[ax_pos0+1][0][1][1]); 
		  pos_plus90[ my_ax_pos0][-tang_pos_num] += (Projall2[ax_pos0][0][1][2]+ Projall2[ax_pos0+1][0][1][2]); 
		  pos_min180[ my_ax_pos0][-tang_pos_num] += ( Projall2[ax_pos0][0][1][3]+ Projall2[ax_pos0+1][0][1][3]); 
		}
	    }   
        }     
        
      }// end of loop over tang_pos_num      
      
    }// end loop over D
  }// end of else
    
  stop_timers();
}

void
ForwardProjectorByBinUsingRayTracing::
 actual_forward_project(Bin& this_bin,
                        const DiscretisedDensity<3,float>& density)
{
    error("ForwardProjectorByBinUsingRayTracing is not supported for list-mode data. Abort.");
}

END_NAMESPACE_STIR
