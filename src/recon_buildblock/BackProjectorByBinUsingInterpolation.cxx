//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup projection

  \brief non-inline implementations for BackProjectorByBinUsingInterpolation

  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project
  
  $Date$
  $Revision$
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/Array.h"
#include "stir/IndexRange4D.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/round.h"
#include "stir/shared_ptr.h"
#include "stir/zoom.h"
#include <memory>
#include <math.h>

#ifndef STIR_NAMESPACES
using std::auto_ptr;
#endif

START_NAMESPACE_STIR

JacobianForIntBP::
JacobianForIntBP(const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, bool exact)
     
     : R2(square(proj_data_info_ptr->get_ring_radius())),
       dxy2(square(proj_data_info_ptr->get_tangential_sampling())),
       ring_spacing2 (square(proj_data_info_ptr->get_ring_spacing())),
       backprojection_normalisation 
      (proj_data_info_ptr->get_ring_spacing()/2/proj_data_info_ptr->get_num_views()),
      use_exact_Jacobian_now(exact)      
   {}

const char * const 
BackProjectorByBinUsingInterpolation::registered_name =
  "Interpolation";


void
BackProjectorByBinUsingInterpolation::
set_defaults()
{
  use_piecewise_linear_interpolation_now = true;
  use_exact_Jacobian_now = true;

  // next can be set to false, but with some rounding error problems (e.g. at 90 degrees)
  do_symmetry_90degrees_min_phi = true;
  // next 2 have to be true, otherwise breaks
  do_symmetry_180degrees_min_phi = true;
  do_symmetry_swap_segment = true;
  // next 2 can be set to false but are ignored anyway
  do_symmetry_swap_s = true;
  do_symmetry_shift_z = true;

}

void
BackProjectorByBinUsingInterpolation::
initialise_keymap()
{
  parser.add_start_key("Back Projector Using Interpolation Parameters");
  parser.add_stop_key("End Back Projector Using Interpolation Parameters");
  parser.add_key("use_piecewise_linear_interpolation", &use_piecewise_linear_interpolation_now);
  parser.add_key("use_exact_Jacobian",&use_exact_Jacobian_now);
#ifdef STIR_DEVEL
  // see set_defaults()
  parser.add_key("do_symmetry_90degrees_min_phi", &do_symmetry_90degrees_min_phi);
  parser.add_key("do_symmetry_180degrees_min_phi", &do_symmetry_180degrees_min_phi);
  parser.add_key("do_symmetry_swap_segment", &do_symmetry_swap_segment);
  parser.add_key("do_symmetry_swap_s", &do_symmetry_swap_s);
  parser.add_key("do_symmetry_shift_z", &do_symmetry_shift_z);
#endif
}

const DataSymmetriesForViewSegmentNumbers *
 BackProjectorByBinUsingInterpolation::get_symmetries_used() const
{ return symmetries_ptr.get(); }

BackProjectorByBinUsingInterpolation::
BackProjectorByBinUsingInterpolation(const bool use_piecewise_linear_interpolation,
                                     const bool use_exact_Jacobian) 
{
  set_defaults();
  use_piecewise_linear_interpolation_now = use_piecewise_linear_interpolation;
  use_exact_Jacobian_now = use_exact_Jacobian;
}

BackProjectorByBinUsingInterpolation::
BackProjectorByBinUsingInterpolation(shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
				     shared_ptr<DiscretisedDensity<3,float> > const& image_info_ptr,
				     const bool use_piecewise_linear_interpolation,
                                     const bool use_exact_Jacobian)
{
  set_defaults();
  use_piecewise_linear_interpolation_now = use_piecewise_linear_interpolation;
  use_exact_Jacobian_now = use_exact_Jacobian;
  set_up(proj_data_info_ptr, image_info_ptr);
}

void
BackProjectorByBinUsingInterpolation::set_up(shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
				     shared_ptr<DiscretisedDensity<3,float> > const& image_info_ptr)
{
  symmetries_ptr = 
    new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr, image_info_ptr,
                                                do_symmetry_90degrees_min_phi,
                                                do_symmetry_180degrees_min_phi,
						do_symmetry_swap_segment,
						do_symmetry_swap_s,
						do_symmetry_shift_z);

   // check if data are according to what we can handle

  const VoxelsOnCartesianGrid<float> * vox_image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (image_info_ptr.get());

  if (vox_image_info_ptr == NULL)
    error("BackProjectorByBinUsingInterpolation initialised with a wrong type of DiscretisedDensity\n");

  const CartesianCoordinate3D<float> voxel_size = vox_image_info_ptr->get_voxel_size();

  // z_origin_in_planes should be an integer
  const float z_origin_in_planes =
    image_info_ptr->get_origin().z()/voxel_size.z();
  if (fabs(round(z_origin_in_planes) - z_origin_in_planes) > 1.E-4)
    error("BackProjectorByBinUsingInterpolation: the shift in the "
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
      error("BackProjectorByBinUsingInterpolation: the number of image planes "
            "per axial_pos (which is %g for segment %d) should be an integer\n",
             num_planes_per_axial_pos, segment_num);
  }
  

}

void
BackProjectorByBinUsingInterpolation::
use_exact_Jacobian(const bool use_exact_Jacobian)
{
  use_exact_Jacobian_now = use_exact_Jacobian;
}


void
BackProjectorByBinUsingInterpolation::
use_piecewise_linear_interpolation(const bool use_piecewise_linear_interpolation)
{
  use_piecewise_linear_interpolation_now = use_piecewise_linear_interpolation;
}

void BackProjectorByBinUsingInterpolation::
actual_back_project(DiscretisedDensity<3,float>& density,
		    const RelatedViewgrams<float>& viewgrams,
		    const int min_axial_pos_num, const int max_axial_pos_num,
		    const int min_tangential_pos_num, const int max_tangential_pos_num)

{
  const ProjDataInfoCylindricalArcCorr* proj_data_info_cyl_ptr = 
    dynamic_cast<const ProjDataInfoCylindricalArcCorr*> (viewgrams.get_proj_data_info_ptr());
 

  if ( proj_data_info_cyl_ptr==NULL)
  {
    error("\nBackProjectorByBinUsingInterpolation:\n"
	  "can only handle arc-corrected data (cast to ProjDataInfoCylindricalArcCorr)!\n");
  }
  // this will throw an exception when the cast does not work
  VoxelsOnCartesianGrid<float>& image = 
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(density);
  // TODO somehow check symmetry object in RelatedViewgrams

  const float zoom = 
    proj_data_info_cyl_ptr->get_tangential_sampling()/
    image.get_voxel_size().x();

  // zoom the viewgrams if necessary
  // if zoom==1 there's no need for allocation of a new
  // RelatedViewgrams object, so we do some trickery with a pointer
  const RelatedViewgrams<float>* zoomed_viewgrams_ptr = 0;
  // to make it exception-proof we need to use an auto_ptr or shared_ptr
  shared_ptr<RelatedViewgrams<float> > zoomed_viewgrams_sptr;
  int zoomed_min_tangential_pos_num;
  int zoomed_max_tangential_pos_num;

  // warning: this criterion has to be the same as the error-check in x,y voxel size
  // (see lines around warning message 'must be equal to'... which occurs more than once)
  if (fabs(zoom-1) > 1E-4)
    {
      zoomed_min_tangential_pos_num = 
	static_cast<int>(ceil(min_tangential_pos_num*zoom));
      zoomed_max_tangential_pos_num = 
	static_cast<int>(ceil(max_tangential_pos_num*zoom));
      // store it in an auto_ptr, such that it gets cleaned up correctly
      zoomed_viewgrams_sptr = 
	new RelatedViewgrams<float>(viewgrams);
      zoomed_viewgrams_ptr = zoomed_viewgrams_sptr.get();

      zoom_viewgrams(*zoomed_viewgrams_sptr, zoom, 
		    zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
    }
  else
    {
      zoomed_min_tangential_pos_num = 
	min_tangential_pos_num;
      zoomed_max_tangential_pos_num = 
	max_tangential_pos_num;
      // we cannot use the auto_ptr here, as that would try to free the 
      // viewgrams object
      zoomed_viewgrams_ptr = &viewgrams;
    }

  const int num_views = viewgrams.get_proj_data_info_ptr()->get_num_views();
  RelatedViewgrams<float>::const_iterator r_viewgrams_iter = zoomed_viewgrams_ptr->begin();
  if (zoomed_viewgrams_ptr->get_basic_segment_num() == 0)
    {
      // no segment symmetry
      const Viewgram<float> & pos_view =*r_viewgrams_iter;
      const Viewgram<float> neg_view = pos_view.get_empty_copy(); 

      if (zoomed_viewgrams_ptr->get_num_viewgrams() == 1)
	{
	  const Viewgram<float> pos_plus90 =  pos_view.get_empty_copy();
	  const Viewgram<float>& neg_plus90 = pos_plus90; 
	  back_project_view_plus_90_and_delta(
					      image,
					      pos_view, neg_view, pos_plus90, neg_plus90, 
					      min_axial_pos_num, max_axial_pos_num,
					      zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
	}
      else
	{
	  r_viewgrams_iter++;
	  if (zoomed_viewgrams_ptr->get_num_viewgrams() == 2)
	    {
	      if (r_viewgrams_iter->get_view_num() == pos_view.get_view_num() + num_views/2)
		{
		  const Viewgram<float> & pos_plus90 =*r_viewgrams_iter;
		  const Viewgram<float> neg_plus90 = pos_plus90.get_empty_copy(); 
		  assert(pos_plus90.get_view_num() == num_views / 2 + pos_view.get_view_num());
		  back_project_view_plus_90_and_delta(
						      image,
						      pos_view, neg_view, pos_plus90, neg_plus90, 
						      min_axial_pos_num, max_axial_pos_num,
						      zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
		}
	      else if (r_viewgrams_iter->get_view_num() == num_views-pos_view.get_view_num())
		{
		  assert(zoomed_viewgrams_ptr->get_basic_view_num() != 0);
		  const Viewgram<float> & pos_min180 =*r_viewgrams_iter;
		  const Viewgram<float> neg_min180 = pos_min180.get_empty_copy(); 
		  const Viewgram<float>& pos_plus90 =neg_min180;// anything 0 really
		  const Viewgram<float>& neg_plus90 = pos_plus90;
		  const Viewgram<float>& pos_min90 =neg_min180;// anything 0 really
		  const Viewgram<float>& neg_min90 = pos_min90;

		  assert(pos_min180.get_view_num() == num_views - pos_view.get_view_num());

		  back_project_all_symmetries(
					      image,
					      pos_view, neg_view, pos_plus90, neg_plus90, 
					      pos_min180, neg_min180, pos_min90, neg_min90,
					      min_axial_pos_num, max_axial_pos_num,
					      zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
		}
	      else
		{
		  error("BackProjectorByBinUsingInterpolation: back_project called with RelatedViewgrams with inconsistent views");
		}
	    }
	  else
	    {
	      assert(zoomed_viewgrams_ptr->get_basic_view_num() != 0);
	      assert(zoomed_viewgrams_ptr->get_basic_view_num() != num_views/4);
	      const Viewgram<float> & pos_plus90 =*r_viewgrams_iter;
	      const Viewgram<float> neg_plus90 = pos_plus90.get_empty_copy(); 
	      r_viewgrams_iter++;//2 
	      const Viewgram<float> & pos_min180 =*r_viewgrams_iter;
	      r_viewgrams_iter++;//3
	      const Viewgram<float> & pos_min90 =*r_viewgrams_iter;
	      const Viewgram<float>& neg_min180 = neg_plus90;//pos_min180.get_empty_copy(); 
	      const Viewgram<float>& neg_min90 = neg_plus90;//pos_min90.get_empty_copy();     

	      assert(pos_plus90.get_view_num() == num_views / 2 + pos_view.get_view_num());
	      assert(pos_min90.get_view_num() == num_views / 2 - pos_view.get_view_num());
	      assert(pos_min180.get_view_num() == num_views - pos_view.get_view_num());

	      back_project_all_symmetries(
					  image,
					  pos_view, neg_view, pos_plus90, neg_plus90, 
					  pos_min180, neg_min180, pos_min90, neg_min90,
					  min_axial_pos_num, max_axial_pos_num,
					  zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
	    }
	}
    }
  else
    {
      // segment symmetry

      if (zoomed_viewgrams_ptr->get_num_viewgrams() == 1)
	error("BackProjectorByBinUsingInterpolation: back_project called with RelatedViewgrams with unexpect number of related viewgrams");

      const Viewgram<float> & pos_view = *r_viewgrams_iter;//0
      r_viewgrams_iter++;
      const Viewgram<float> & neg_view = *r_viewgrams_iter;//1
      assert(neg_view.get_view_num() == pos_view.get_view_num());
	   
      if (zoomed_viewgrams_ptr->get_num_viewgrams() == 2)
	{
	  const Viewgram<float> pos_plus90 =  pos_view.get_empty_copy();
	  const Viewgram<float>& neg_plus90 = pos_plus90; 
	  back_project_view_plus_90_and_delta(
					      image,
					      pos_view, neg_view, pos_plus90, neg_plus90, 
					      min_axial_pos_num, max_axial_pos_num,
					      zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
	}
      else if (zoomed_viewgrams_ptr->get_num_viewgrams() == 4)
	{
	  r_viewgrams_iter++;
      
	  if (r_viewgrams_iter->get_view_num() == pos_view.get_view_num() + num_views/2)
	    {
	      const Viewgram<float> & pos_plus90 =*r_viewgrams_iter;//2
	      r_viewgrams_iter++;
	      const Viewgram<float> & neg_plus90 =*r_viewgrams_iter;//3

	      assert(pos_plus90.get_view_num() == num_views / 2 + pos_view.get_view_num());
	      assert(neg_plus90.get_view_num() == num_views / 2 + pos_view.get_view_num());
	      back_project_view_plus_90_and_delta(
						  image,
						  pos_view, neg_view, pos_plus90, neg_plus90, 
						  min_axial_pos_num, max_axial_pos_num,
						  zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
	    }
	  else if (r_viewgrams_iter->get_view_num() == num_views-pos_view.get_view_num())
	    {
	      assert(zoomed_viewgrams_ptr->get_basic_view_num() != 0);
	      const Viewgram<float> & pos_min180 =*r_viewgrams_iter; //2
	      r_viewgrams_iter++;
	      const Viewgram<float> & neg_min180 =*r_viewgrams_iter;//3
	      const Viewgram<float>& pos_plus90 =pos_view.get_empty_copy();// anything 0 really
	      const Viewgram<float>& neg_plus90 = pos_plus90;
	      const Viewgram<float>& pos_min90 = pos_plus90;
	      const Viewgram<float>& neg_min90 = pos_plus90;

	      assert(pos_min180.get_view_num() == num_views - pos_view.get_view_num());
	      assert(neg_min180.get_view_num() == num_views - pos_view.get_view_num());

	      back_project_all_symmetries(
					  image,
					  pos_view, neg_view, pos_plus90, neg_plus90, 
					  pos_min180, neg_min180, pos_min90, neg_min90,
					  min_axial_pos_num, max_axial_pos_num,
					      zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);
	    }
	  else
	    {
	      error("BackProjectorByBinUsingInterpolation: back_project called with RelatedViewgrams with inconsistent views");
	    }

	}
      else  if (zoomed_viewgrams_ptr->get_num_viewgrams() == 8)
	{
	  assert(zoomed_viewgrams_ptr->get_basic_view_num() != 0);
	  assert(zoomed_viewgrams_ptr->get_basic_view_num() != num_views/4);
	  r_viewgrams_iter++;
	  const Viewgram<float> & pos_plus90 =*r_viewgrams_iter;//2
	  r_viewgrams_iter++;
	  const Viewgram<float> & neg_plus90 =*r_viewgrams_iter;//3
	  r_viewgrams_iter++;//4
	  const Viewgram<float> & pos_min180 =*r_viewgrams_iter;
	  r_viewgrams_iter++;//5
	  const Viewgram<float> & neg_min180 =*r_viewgrams_iter;
	  r_viewgrams_iter++;//6
	  const Viewgram<float> & pos_min90 =*r_viewgrams_iter;
	  r_viewgrams_iter++;//7
	  const Viewgram<float> & neg_min90 =*r_viewgrams_iter;

	  assert(pos_plus90.get_view_num() == num_views / 2 + pos_view.get_view_num());
	  assert(pos_min90.get_view_num() == num_views / 2 - pos_view.get_view_num());
	  assert(pos_min180.get_view_num() == num_views - pos_view.get_view_num());
	  assert(neg_view.get_view_num() == pos_view.get_view_num());
	  assert(neg_plus90.get_view_num() == pos_plus90.get_view_num());
	  assert(neg_min90.get_view_num() == pos_min90.get_view_num());
	  assert(neg_min180.get_view_num() == pos_min180.get_view_num());

	  back_project_all_symmetries(
				      image,
				      pos_view, neg_view, pos_plus90, neg_plus90, 
				      pos_min180, neg_min180, pos_min90, neg_min90,
				      min_axial_pos_num, max_axial_pos_num,
				      zoomed_min_tangential_pos_num, zoomed_max_tangential_pos_num);

	}
    }

}





#if 0
/******************************************************************************
 2D
 ******************************************************************************/

// TODO rounding errors...

void 
BackProjectorByBinUsingInterpolation::
  back_project_2D_all_symmetries(const Sinogram<float> &sino, PETPlane &image, int view,
                                    const int min_tang_pos, const int max_tang_pos)
{
  start_timers();

  assert(sino.get_min_bin() == - sino.get_max_bin());
  assert(min_tang_pos == -max_tang_pos);
  assert(image.get_min_x() == - image.get_max_x());
  assert(image.get_min_y() == - image.get_max_y());
  assert(view < sino.get_num_views() / 4);
  
  assert(view != 0);
  
  const int nviews = sino.get_num_views();
  const int view90=(int)(nviews/2);
  const JacobianForIntBP jacobian(sino.scan_info, use_exact_Jacobian_now);
  
  
  const int        min90 = view90 - view;
  const int        plus90 = view90 + view;
  const int        min180 = nviews - view;
  
  const double        phi = _PI * view / nviews;
  // sadly has to be float, otherwise rounding errors...
  const float        cphi = cos(phi);
  const float        sphi = sin(phi);
  
  
  ProjDataForIntBP projs;
  //TODO loop is wrong
  for (int s = 0; s <= max_tang_pos - 2; s++)
  {
    const float jac = jacobian(0, s+ 0.5);
    
    projs.view__pos_s = sino[view][s] * jac;
    projs.view__pos_sp1 =sino[view][s+1] * jac;
    projs.view__neg_s = sino[view][-s] * jac; 
    projs.view__neg_sp1 = sino[view][-s-1] * jac;
    
    projs.min90__pos_s = sino[min90][s] * jac; 
    projs.min90__pos_sp1 =sino[min90][s+1] * jac;
    projs.min90__neg_s = sino[min90][-s] * jac; 
    projs.min90__neg_sp1 = sino[min90][-s-1] * jac;
    
    projs.plus90__pos_s = sino[plus90][s] * jac; 
    projs.plus90__pos_sp1 =sino[plus90][s+1] * jac;
    projs.plus90__neg_s = sino[plus90][-s] * jac; 
    projs.plus90__neg_sp1 = sino[plus90][-s-1] * jac;
    
    projs.min180__pos_s = sino[min180][s] * jac; 
    projs.min180__pos_sp1 =sino[min180][s+1] * jac;
    projs.min180__neg_s = sino[min180][-s] * jac; 
    projs.min180__neg_sp1 = sino[min180][-s-1] * jac;
    
    backproj2D_Cho_view_viewplus90_180minview_90minview(image, projs, cphi, sphi, s);
  }
  
  stop_timers();
}

void 
BackProjectorByBinUsingInterpolation::
  back_project_2D_view_plus_90(const Sinogram<float> &sino, PETPlane &image, int view,
                               const int min_tang_pos, const int max_tang_pos)
{
  start_timers();

  assert(sino.get_min_bin() == - sino.get_max_bin());
  assert(min_tang_pos == -max_tang_pos);
  assert(image.get_min_x() == - image.get_max_x());
  assert(image.get_min_y() == - image.get_max_y());
  assert(view < sino.get_num_views() / 2);
  
  assert(view >= 0);
  assert(view < sino.get_num_views()/2);
  
  const int nviews = sino.get_num_views();
  const int view90 = nviews/2;
  const JacobianForIntBP jacobian(sino.scan_info, use_exact_Jacobian_now);
  
  
  const int        min90 = view90 - view;
  const int        plus90 = view90 + view;
  const int        min180 = nviews - view;
  
  const double        phi = _PI * view / nviews;
  const float        cphi = cos(phi);
  const float        sphi = sin(phi);
  
  
  ProjDataForIntBP projs;
  // TODO loop is wrong
  for (int s = 0; s <= max_tang_pos - 2; s++)
  {
    const float jac = jacobian(0, s+ 0.5);
    
    projs.view__pos_s = sino[view][s] * jac;
    projs.view__pos_sp1 =sino[view][s+1] * jac;
    projs.view__neg_s = sino[view][-s] * jac; 
    projs.view__neg_sp1 = sino[view][-s-1] * jac;
    
    projs.plus90__pos_s = sino[plus90][s] * jac; 
    projs.plus90__pos_sp1 =sino[plus90][s+1] * jac;
    projs.plus90__neg_s = sino[plus90][-s] * jac; 
    projs.plus90__neg_sp1 = sino[plus90][-s-1] * jac;
        
    backproj2D_Cho_view_viewplus90(image, projs, cphi, sphi, s);
  }
 
  stop_timers();
}

#endif



/****************************************************************************
 real work
 ****************************************************************************/
   /*
    The version which uses all possible symmetries.
    Here 0<=view < num_views/4 (= 45 degrees)
    */

void 
BackProjectorByBinUsingInterpolation::back_project_all_symmetries(
				 VoxelsOnCartesianGrid<float>& image,
			    	 const Viewgram<float> & pos_view, 
				 const Viewgram<float> & neg_view, 
				 const Viewgram<float> & pos_plus90, 
				 const Viewgram<float> & neg_plus90, 
				 const Viewgram<float> & pos_min180, 
				 const Viewgram<float> & neg_min180, 
				 const Viewgram<float> & pos_min90, 
				 const Viewgram<float> & neg_min90,
				 const int min_axial_pos_num, const int max_axial_pos_num,
				 const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  const ProjDataInfoCylindricalArcCorr* proj_data_info_cyl_ptr = 
    dynamic_cast<const ProjDataInfoCylindricalArcCorr*> (pos_view.get_proj_data_info_ptr());
 

  if ( proj_data_info_cyl_ptr==NULL)
  {
    error("\nBackProjectorByBinUsingInterpolation:\n\
can only handle arc-corrected data (cast to ProjDataInfoCylindricalArcCorr)!\n");
  }

  assert(min_axial_pos_num >= pos_view. get_min_axial_pos_num());
  assert(max_axial_pos_num <= pos_view. get_max_axial_pos_num());
  assert(min_tangential_pos_num >= pos_view.get_min_tangential_pos_num());
  assert(max_tangential_pos_num <= pos_view.get_max_tangential_pos_num());

  //KTxxx not necessary anymore
  //assert(min_tangential_pos_num == - max_tangential_pos_num);

#ifndef NDEBUG
  // This variable is only used in assert() at the moment, so avoid compiler 
  // warning by defining it only when in debug mode
  const int segment_num = pos_view.get_segment_num();
#endif

  
  assert(proj_data_info_cyl_ptr ->get_average_ring_difference(segment_num) >= 0);
  assert(pos_view.get_view_num() > 0);
  assert(pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/4 ||
	 (!symmetries_ptr->using_symmetry_90degrees_min_phi() &&
	  pos_view.get_view_num() < pos_view.get_proj_data_info_ptr()->get_num_views()/2 &&
	  pos_plus90.find_max()==0 && neg_plus90.find_max()==0 && 
	  pos_min90.find_max()==0 && neg_min90.find_max()==0) );

  const int nviews = pos_view.get_proj_data_info_ptr()->get_num_views();

  // warning: error check has to be the same as what is used for the criterion to do the zooming
  // (see lines concerning zoomed_viewgrams)
  if(fabs(image.get_voxel_size().x()/proj_data_info_cyl_ptr->get_tangential_sampling() - 1) > 1E-4
     || fabs(image.get_voxel_size().y()/proj_data_info_cyl_ptr->get_tangential_sampling() - 1) > 1E-4)
    error("BackProjectorByBinUsingInterpolation: x,y voxel size must be equal to bin size.");
      
  // KTxxx not necessary anymore
  //assert(image.get_min_z() == 0);

  if (pos_view.get_view_num() == 0)
    {
      error("BackProjectorByBinUsingInterpolation: back_project_all_symmetries called with view 0 degrees.\n");
    }
  if (symmetries_ptr->using_symmetry_90degrees_min_phi() &&
       pos_view.get_view_num() == nviews/4)
    {
      error("BackProjectorByBinUsingInterpolation: back_project_all_symmetries called with view 45 degrees.\n");
    }

  // KT XXX
  const float fovrad_in_mm   = 
    min((min(image.get_max_x(), -image.get_min_x()))*image.get_voxel_size().x(),
	(min(image.get_max_y(), -image.get_min_y()))*image.get_voxel_size().y()); 
  const int fovrad = round(fovrad_in_mm/image.get_voxel_size().x());
  // TODO remove -2, it's there because otherwise find_start_values() goes crazy.
  const int max_tang_pos_to_use =
    min(max_tangential_pos_num, fovrad-2);
  const int min_tang_pos_to_use =
    max(min_tangential_pos_num, -(fovrad-2));

  const int max_abs_tang_pos_to_use = 
    max(max_tang_pos_to_use, -min_tang_pos_to_use);
  const int min_abs_tang_pos_to_use = 
    max_tang_pos_to_use<0 ?
      -max_tang_pos_to_use
    : (min_tang_pos_to_use>0 ?
       min_tang_pos_to_use
       : 0 );


  start_timers();

  const JacobianForIntBP jacobian(proj_data_info_cyl_ptr, use_exact_Jacobian_now);

  Array<4, float > Proj2424(IndexRange4D(0, 1, 0, 3, 0, 1, 1, 4));
  // a variable which will be used in the loops over s to get s_in_mm
  Bin bin(pos_view.get_segment_num(), pos_view.get_view_num(),min_axial_pos_num,0);    

  // KT 20/06/2001 rewrite using get_phi  
  const float cphi = cos(proj_data_info_cyl_ptr->get_phi(bin));
  const float sphi = sin(proj_data_info_cyl_ptr->get_phi(bin));
 

  // Do a loop over all axial positions. However, because we use interpolation of
  // a 'beam', each step takes elements from ax_pos and ax_pos+1. So, data in
  // a ring influences beam ax_pos-1 and ax_pos. All this means that we
  // have to let ax_pos run from min_axial_pos_num-1 to max_axial_pos_num.
  for (int ax_pos = min_axial_pos_num-1; ax_pos <= max_axial_pos_num; ax_pos++)
    {
      const int ax_pos_plus = ax_pos + 1; 

      // We have to fill with 0, as not all elements are set in the lines below
      if (ax_pos==min_axial_pos_num-1 || ax_pos==max_axial_pos_num)
	Proj2424.fill(0);
      
      for (int s = min_abs_tang_pos_to_use; s <= max_abs_tang_pos_to_use; s++) {
	const int splus = s + 1;
	const int ms = -s;
	const int msplus = -splus;

	// now I have to check if ax_pos is in allowable range
        if (ax_pos >= min_axial_pos_num)
	{
	Proj2424[0][0][0][1] = s>max_tang_pos_to_use ? 0 : pos_view[ax_pos][s];
	Proj2424[0][0][0][2] = splus>max_tang_pos_to_use ? 0 : pos_view[ax_pos][splus];
	Proj2424[0][1][0][3] = s>max_tang_pos_to_use ? 0 : pos_min90[ax_pos][s];
	Proj2424[0][1][0][4] = splus>max_tang_pos_to_use ? 0 : pos_min90[ax_pos][splus];
	Proj2424[0][2][0][1] = s>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos][s];
	Proj2424[0][2][0][2] = splus>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos][splus];
	Proj2424[0][3][0][3] = s>max_tang_pos_to_use ? 0 : pos_min180[ax_pos][s];
	Proj2424[0][3][0][4] = splus>max_tang_pos_to_use ? 0 : pos_min180[ax_pos][splus];
	Proj2424[1][0][0][3] = s>max_tang_pos_to_use ? 0 : neg_view[ax_pos][s];
	Proj2424[1][0][0][4] = splus>max_tang_pos_to_use ? 0 : neg_view[ax_pos][splus];
	Proj2424[1][1][0][1] = s>max_tang_pos_to_use ? 0 : neg_min90[ax_pos][s];
	Proj2424[1][1][0][2] = splus>max_tang_pos_to_use ? 0 : neg_min90[ax_pos][splus];
	Proj2424[1][2][0][3] = s>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos][s];
	Proj2424[1][2][0][4] = splus>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos][splus];
	Proj2424[1][3][0][1] = s>max_tang_pos_to_use ? 0 : neg_min180[ax_pos][s];
	Proj2424[1][3][0][2] = splus>max_tang_pos_to_use ? 0 : neg_min180[ax_pos][splus];

	Proj2424[0][0][1][3] = ms<min_tang_pos_to_use ? 0 : pos_view[ax_pos][ms];
	Proj2424[0][0][1][4] = msplus<min_tang_pos_to_use ? 0 : pos_view[ax_pos][msplus];
	Proj2424[0][1][1][1] = ms<min_tang_pos_to_use ? 0 : pos_min90[ax_pos][ms];
	Proj2424[0][1][1][2] = msplus<min_tang_pos_to_use ? 0 : pos_min90[ax_pos][msplus];
	Proj2424[0][2][1][3] = ms<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos][ms];
	Proj2424[0][2][1][4] = msplus<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos][msplus];
	Proj2424[0][3][1][1] = ms<min_tang_pos_to_use ? 0 : pos_min180[ax_pos][ms];
	Proj2424[0][3][1][2] = msplus<min_tang_pos_to_use ? 0 : pos_min180[ax_pos][msplus];
	Proj2424[1][0][1][1] = ms<min_tang_pos_to_use ? 0 : neg_view[ax_pos][ms];
	Proj2424[1][0][1][2] = msplus<min_tang_pos_to_use ? 0 : neg_view[ax_pos][msplus];
	Proj2424[1][1][1][3] = ms<min_tang_pos_to_use ? 0 : neg_min90[ax_pos][ms];
	Proj2424[1][1][1][4] = msplus<min_tang_pos_to_use ? 0 : neg_min90[ax_pos][msplus];
	Proj2424[1][2][1][1] = ms<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos][ms];
	Proj2424[1][2][1][2] = msplus<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos][msplus];
	Proj2424[1][3][1][3] = ms<min_tang_pos_to_use ? 0 : neg_min180[ax_pos][ms];
	Proj2424[1][3][1][4] = msplus<min_tang_pos_to_use ? 0 : neg_min180[ax_pos][msplus];
	}

	if (ax_pos_plus <= max_axial_pos_num)
	{
	Proj2424[0][0][0][3] = s>max_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][s];
	Proj2424[0][0][0][4] = splus>max_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][splus];
	Proj2424[0][1][0][1] = s>max_tang_pos_to_use ? 0 : pos_min90[ax_pos_plus][s];
	Proj2424[0][1][0][2] = splus>max_tang_pos_to_use ? 0 : pos_min90[ax_pos_plus][splus];
        Proj2424[0][2][0][3] = s>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][s];
	Proj2424[0][2][0][4] = splus>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][splus];
	Proj2424[0][3][0][1] = s>max_tang_pos_to_use ? 0 : pos_min180[ax_pos_plus][s];
	Proj2424[0][3][0][2] = splus>max_tang_pos_to_use ? 0 : pos_min180[ax_pos_plus][splus];
	Proj2424[1][0][0][1] = s>max_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][s];
	Proj2424[1][0][0][2] = splus>max_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][splus];
	Proj2424[1][1][0][3] = s>max_tang_pos_to_use ? 0 : neg_min90[ax_pos_plus][s];
	Proj2424[1][1][0][4] = splus>max_tang_pos_to_use ? 0 : neg_min90[ax_pos_plus][splus];
	Proj2424[1][2][0][1] = s>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][s];
	Proj2424[1][2][0][2] = splus>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][splus];
	Proj2424[1][3][0][3] = s>max_tang_pos_to_use ? 0 : neg_min180[ax_pos_plus][s];
	Proj2424[1][3][0][4] = splus>max_tang_pos_to_use ? 0 : neg_min180[ax_pos_plus][splus];

	Proj2424[0][0][1][1] = ms<min_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][ms];
	Proj2424[0][0][1][2] = msplus<min_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][msplus];
	Proj2424[0][1][1][3] = ms<min_tang_pos_to_use ? 0 : pos_min90[ax_pos_plus][ms];
	Proj2424[0][1][1][4] = msplus<min_tang_pos_to_use ? 0 : pos_min90[ax_pos_plus][msplus];
	Proj2424[0][2][1][1] = ms<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][ms];
	Proj2424[0][2][1][2] = msplus<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][msplus];
	Proj2424[0][3][1][3] = ms<min_tang_pos_to_use ? 0 : pos_min180[ax_pos_plus][ms];
	Proj2424[0][3][1][4] = msplus<min_tang_pos_to_use ? 0 : pos_min180[ax_pos_plus][msplus];
	Proj2424[1][0][1][3] = ms<min_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][ms];
	Proj2424[1][0][1][4] = msplus<min_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][msplus];
	Proj2424[1][1][1][1] = ms<min_tang_pos_to_use ? 0 : neg_min90[ax_pos_plus][ms];
	Proj2424[1][1][1][2] = msplus<min_tang_pos_to_use ? 0 : neg_min90[ax_pos_plus][msplus];
	Proj2424[1][2][1][3] = ms<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][ms];
	Proj2424[1][2][1][4] = msplus<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][msplus];
	Proj2424[1][3][1][1] = ms<min_tang_pos_to_use ? 0 : neg_min180[ax_pos_plus][ms];
	Proj2424[1][3][1][2] = msplus<min_tang_pos_to_use ? 0 : neg_min180[ax_pos_plus][msplus];
	}
	const int segment_num = pos_view.get_segment_num();

        const float delta=proj_data_info_cyl_ptr->get_average_ring_difference(segment_num);

        // take s+.5 as average for the beam (it's slowly varying in s anyway)
        Proj2424 *= jacobian(delta, s+ 0.5F);

	// find correspondence between ax_pos coordinates and image coordinates:
	// z = num_planes_per_axial_pos * ring + axial_pos_to_z_offset
	// KT 20/06/2001 rewrote using symmetries_ptr
	const int num_planes_per_axial_pos =
          round(symmetries_ptr->get_num_planes_per_axial_pos(segment_num));
	const float axial_pos_to_z_offset = 
	  symmetries_ptr->get_axial_pos_to_z_offset(segment_num);

        if (use_piecewise_linear_interpolation_now && num_planes_per_axial_pos>1)
          piecewise_linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview
          (Proj2424,
          image,
          proj_data_info_cyl_ptr, 
          delta, 
          cphi, sphi, s, ax_pos, 
          num_planes_per_axial_pos,
          axial_pos_to_z_offset);
        else
          linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview
          (Proj2424,
          image,
          proj_data_info_cyl_ptr, 
          delta, 
          cphi, sphi, s, ax_pos, 
          num_planes_per_axial_pos,
          axial_pos_to_z_offset);
      }
    }
  stop_timers();
}

/*
This function projects 4 viewgrams related by symmetry.
It will be used for view=0 or 45 degrees 
(or others if the 90degrees_min_phi symmetry is not used).

Here 0<=view < num_views/2 (= 90 degrees)
*/
void 
BackProjectorByBinUsingInterpolation::
back_project_view_plus_90_and_delta(
				         VoxelsOnCartesianGrid<float>& image,
					 const Viewgram<float> & pos_view, 
					 const Viewgram<float> & neg_view, 
					 const Viewgram<float> & pos_plus90, 
					 const Viewgram<float> & neg_plus90,
					 const int min_axial_pos_num, 
					 const int max_axial_pos_num,
				         const int min_tangential_pos_num, 
					 const int max_tangential_pos_num)				   
{
  const ProjDataInfoCylindricalArcCorr* proj_data_info_cyl_ptr = 
    dynamic_cast<const ProjDataInfoCylindricalArcCorr*> (pos_view.get_proj_data_info_ptr());

  if ( proj_data_info_cyl_ptr==NULL)
    {
      error("\nBackProjectorByBinUsingInterpolation:,\n\
can only handle arc-corrected data (cast to ProjDataInfoCylindricalArcCorr)!\n");
    }

  assert(min_axial_pos_num >= pos_view. get_min_axial_pos_num());
  assert(max_axial_pos_num <= pos_view. get_max_axial_pos_num());
  assert(min_tangential_pos_num >= pos_view.get_min_tangential_pos_num());
  assert(max_tangential_pos_num <= pos_view.get_max_tangential_pos_num());

  // KTXXX not necessary anymore
  //assert(min_tangential_pos_num == - max_tangential_pos_num);
#ifndef NDEBUG
  // These variables are only used in assert() at the moment, so avoid compiler 
  // warning by defining it only when in debug mode
  const int segment_num = pos_view.get_segment_num();
#endif

  assert(proj_data_info_cyl_ptr ->get_average_ring_difference(segment_num) >= 0);

  const int num_views =  pos_view.get_proj_data_info_ptr()->get_num_views();

  assert(pos_view.get_view_num()>=0);
  assert(pos_view.get_view_num() <num_views/2 ||
	 (pos_view.get_view_num() <num_views &&
	  pos_plus90.find_max()==0 && neg_plus90.find_max()==0) );


  // warning: error check has to be the same as what is used for the criterion to do the zooming
  // (see lines concerning zoomed_viewgrams)
  if(fabs(image.get_voxel_size().x()/proj_data_info_cyl_ptr->get_tangential_sampling() - 1) > 1E-4
     || fabs(image.get_voxel_size().y()/proj_data_info_cyl_ptr->get_tangential_sampling() - 1) > 1E-4)
    error("BackProjectorByBinUsingInterpolation: x,y voxel size must be equal to bin size.");

  // KTXXX not necessary anymore
  //assert(image.get_min_z() == 0);

  start_timers();

  const JacobianForIntBP jacobian(proj_data_info_cyl_ptr, use_exact_Jacobian_now);

  Array<4, float > Proj2424(IndexRange4D(0, 1, 0, 3, 0, 1, 1, 4));

  // a variable which will be used in the loops over s to get s_in_mm
  Bin bin(pos_view.get_segment_num(), pos_view.get_view_num(),min_axial_pos_num,0);    

  // compute cos(phi) and sin(phi)
  /* KT included special cases for phi=0 and 90 degrees to try to avoid rounding problems
    This is because the current low-level code has problems with e.g. cos(phi) being
    a very small negative number.*/
  const float cphi = 
   bin.view_num()==0? 1 :
   2*bin.view_num()==num_views ? 0 :
   cos(proj_data_info_cyl_ptr->get_phi(bin));
  const float sphi = 
   bin.view_num()==0? 0 :
   2*bin.view_num()==num_views ? 1 :
  sin(proj_data_info_cyl_ptr->get_phi(bin));
 
  assert(fabs(cphi-cos(proj_data_info_cyl_ptr->get_phi(bin)))<.001);
  assert(fabs(sphi-sin(proj_data_info_cyl_ptr->get_phi(bin)))<.001);
  // KT XXX
  const float fovrad_in_mm   = 
    min((min(image.get_max_x(), -image.get_min_x()))*image.get_voxel_size().x(),
	(min(image.get_max_y(), -image.get_min_y()))*image.get_voxel_size().y()); 
  const int fovrad = round(fovrad_in_mm/image.get_voxel_size().x());
  // TODO remove -2, it's there because otherwise find_start_values() goes crazy.
  const int max_tang_pos_to_use =
    min(max_tangential_pos_num, fovrad-2);
  const int min_tang_pos_to_use =
    max(min_tangential_pos_num, -(fovrad-2));


  const int max_abs_tang_pos_to_use = 
    max(max_tang_pos_to_use, -min_tang_pos_to_use);
  const int min_abs_tang_pos_to_use = 
    max_tang_pos_to_use<0 ?
      -max_tang_pos_to_use
    : (min_tang_pos_to_use>0 ?
       min_tang_pos_to_use
       : 0 );

  // Do a loop over all axial positions. However, because we use interpolation of
  // a 'beam', each step takes elements from ax_pos and ax_pos+1. So, data at
  // ax_pos influences beam ax_pos-1 and ax_pos. All this means that we
  // have to let ax_pos run from min_axial_pos_num-1 to max_axial_pos_num.
  for (int ax_pos = min_axial_pos_num-1; ax_pos <= max_axial_pos_num; ax_pos++)
    {
      const int ax_pos_plus = ax_pos + 1; 
        
      // We have to fill with 0, as not all elements are set in the lines below
      if (ax_pos==min_axial_pos_num-1 || ax_pos==max_axial_pos_num)
	Proj2424.fill(0);
      for (int s = min_abs_tang_pos_to_use; s <= max_abs_tang_pos_to_use; s++) {
	const int splus = s + 1;
	const int ms = -s;
	const int msplus = -splus;

	// now I have to check if ax_pos is in allowable range
	if (ax_pos >= min_axial_pos_num)
	  {
	    Proj2424[0][0][0][1] = s>max_tang_pos_to_use ? 0 : pos_view[ax_pos][s];
	    Proj2424[0][0][0][2] = splus>max_tang_pos_to_use ? 0 : pos_view[ax_pos][splus];
	    Proj2424[0][2][0][1] = s>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos][s];
	    Proj2424[0][2][0][2] = splus>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos][splus];
	    Proj2424[1][0][0][3] = s>max_tang_pos_to_use ? 0 : neg_view[ax_pos][s];
	    Proj2424[1][0][0][4] = splus>max_tang_pos_to_use ? 0 : neg_view[ax_pos][splus];
	    Proj2424[1][2][0][3] = s>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos][s];
	    Proj2424[1][2][0][4] = splus>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos][splus];

	    Proj2424[0][0][1][3] = ms<min_tang_pos_to_use ? 0 : pos_view[ax_pos][ms];
	    Proj2424[0][0][1][4] = msplus<min_tang_pos_to_use ? 0 : pos_view[ax_pos][msplus];
	    Proj2424[0][2][1][3] = ms<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos][ms];
	    Proj2424[0][2][1][4] = msplus<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos][msplus];
	    Proj2424[1][0][1][1] = ms<min_tang_pos_to_use ? 0 : neg_view[ax_pos][ms];
	    Proj2424[1][0][1][2] = msplus<min_tang_pos_to_use ? 0 : neg_view[ax_pos][msplus];
	    Proj2424[1][2][1][1] = ms<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos][ms];
	    Proj2424[1][2][1][2] = msplus<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos][msplus];
	  }

        if (ax_pos_plus <= max_axial_pos_num)
	  {
	    Proj2424[0][0][0][3] = s>max_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][s];
	    Proj2424[0][0][0][4] = splus>max_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][splus];
	    Proj2424[0][2][0][3] = s>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][s];
	    Proj2424[0][2][0][4] = splus>max_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][splus];
	    Proj2424[1][0][0][1] = s>max_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][s];
	    Proj2424[1][0][0][2] = splus>max_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][splus];
	    Proj2424[1][2][0][1] = s>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][s];
	    Proj2424[1][2][0][2] = splus>max_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][splus];

	    Proj2424[0][0][1][1] = ms<min_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][ms];
	    Proj2424[0][0][1][2] = msplus<min_tang_pos_to_use ? 0 : pos_view[ax_pos_plus][msplus];
	    Proj2424[0][2][1][1] = ms<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][ms];
	    Proj2424[0][2][1][2] = msplus<min_tang_pos_to_use ? 0 : pos_plus90[ax_pos_plus][msplus];
	    Proj2424[1][0][1][3] = ms<min_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][ms];
	    Proj2424[1][0][1][4] = msplus<min_tang_pos_to_use ? 0 : neg_view[ax_pos_plus][msplus];
	    Proj2424[1][2][1][3] = ms<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][ms];
	    Proj2424[1][2][1][4] = msplus<min_tang_pos_to_use ? 0 : neg_plus90[ax_pos_plus][msplus];
	  }

	const int segment_num = pos_view.get_segment_num();
	const float delta=proj_data_info_cyl_ptr->get_average_ring_difference(segment_num);

        // take s+.5 as average for the beam (it's slowly varying in s anyway)
        Proj2424 *= jacobian(delta, s+ 0.5F);
        
	// find correspondence between ax_pos coordinates and image coordinates:
	// z = num_planes_per_axial_pos * ring + axial_pos_to_z_offset
	// KT 20/06/2001 rewrote using symmetries_ptr
	const int num_planes_per_axial_pos =
          round(symmetries_ptr->get_num_planes_per_axial_pos(segment_num));
	const float axial_pos_to_z_offset = 
	  symmetries_ptr->get_axial_pos_to_z_offset(segment_num);

        if (use_piecewise_linear_interpolation_now && num_planes_per_axial_pos>1)
          piecewise_linear_interpolation_backproj3D_Cho_view_viewplus90( Proj2424, image, 
									 proj_data_info_cyl_ptr, 
									 delta, 
									 cphi, sphi, s, ax_pos, 
									 num_planes_per_axial_pos,
									 axial_pos_to_z_offset);
        else
          linear_interpolation_backproj3D_Cho_view_viewplus90( Proj2424, image, 
							       proj_data_info_cyl_ptr, 
							       delta, 
							       cphi, sphi, s, ax_pos, 
							       num_planes_per_axial_pos,
							       axial_pos_to_z_offset);
      }
    }
  stop_timers();
}


END_NAMESPACE_STIR
