//
// $Id$
//
/*!
  \file
  \ingroup FBP2D

  \brief Implementation of class FBP2DReconstruction

  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project

  $Date$
  $Revision$
*/
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

#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/analytic/FBP2D/RampFilter.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Bin.h"
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
START_NAMESPACE_STIR


void 
FBP2DReconstruction::
set_defaults()
{
  Reconstruction::set_defaults();

  alpha_ramp = 1;
  fc_ramp = 0.5;
  pad_in_s=1;
  display_level=0; // no display
  num_segments_to_combine = -1;
  back_projector_sptr =
    new BackProjectorByBinUsingInterpolation(
					     /*use_piecewise_linear_interpolation = */true, 
					     /*use_exact_Jacobian = */ false);
 
}

void 
FBP2DReconstruction::initialise_keymap()
{
  Reconstruction::initialise_keymap();

  parser.add_start_key("FBP2DParameters");
  parser.add_stop_key("End");
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
  parser.add_key("Alpha parameter for Ramp filter",  &alpha_ramp);
  parser.add_key("Cut-off for Ramp filter (in cycles)",&fc_ramp);
  parser.add_key("Transaxial extension for FFT", &pad_in_s);
  parser.add_key("Display level",&display_level);

  parser.add_parsing_key("Back projector type", &back_projector_sptr);
}

void 
FBP2DReconstruction::
ask_parameters()
{ 
   
  Reconstruction::ask_parameters();

  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)",-1,101,-1);
  alpha_ramp =  ask_num(" Alpha parameter for Ramp filter ? ",0.,1., 1.);    
  fc_ramp =  ask_num(" Cut-off frequency for Ramp filter ? ",0.,.5, 0.5);
  pad_in_s = ask_num("  Transaxial extension for FFT : ",0,1, 1); 
  display_level = ask_num("Which images would you like to display \n\t(0: None, 1: Final, 2: filtered viewgrams) ? ", 0,2,0);

#if 0
    // do not ask the user for the projectors to prevent them entering
    // silly things
  do 
    {
      back_projector_sptr =
	BackProjectorByBin::ask_type_and_parameters();
    }
#endif

}

bool FBP2DReconstruction::post_processing()
{
  if (Reconstruction::post_processing())
    return true;
  return post_processing_only_FBP2D_parameters();
}

bool FBP2DReconstruction::post_processing_only_FBP2D_parameters()
{
  if (fc_ramp<=0 || fc_ramp>.5000000001)
    {
      warning("Cut-off frequency has to be between 0 and .5 but is %g\n", fc_ramp);
      return true;
    }
  if (alpha_ramp<=0 || alpha_ramp>1.000000001)
    {
      warning("Alpha parameter for ramp has to be between 0 and 1 but is %g\n", alpha_ramp);
      return true;
    }
  if (pad_in_s<0 || pad_in_s>2)
    {
      warning("padding factor has to be between 0 and 2 but is %d\n", pad_in_s);
      return true;
    }
  if (pad_in_s<1)
      warning("Transaxial extension for FFT:=0 should ONLY be used when the non-zero data\n"
	      "occupy only half of the FOV. Otherwise aliasing will occur!");

  if (num_segments_to_combine>=0 && num_segments_to_combine%2==0)
    {
      warning("num_segments_to_combine has to be odd (or -1), but is %d\n", num_segments_to_combine);
      return true;
    }

    if (num_segments_to_combine==-1)
    {
      const ProjDataInfoCylindrical * proj_data_info_cyl_ptr =
	dynamic_cast<const ProjDataInfoCylindrical *>(proj_data_ptr->get_proj_data_info_ptr());

      if (proj_data_info_cyl_ptr==0)
        num_segments_to_combine = 1;
      else
	{
	  if (proj_data_info_cyl_ptr->get_min_ring_difference(0) != 
	      proj_data_info_cyl_ptr->get_max_ring_difference(0))
	    num_segments_to_combine = 1;
	  else
	    num_segments_to_combine = 3;
	}
    }

    if (is_null_ptr(back_projector_sptr))
      {
	warning("Back projector not set.\n");
	return true;
      }

  return false;
}

string FBP2DReconstruction::method_info() const
{
  return "FBP2D";
}

FBP2DReconstruction::
FBP2DReconstruction(const string& parameter_filename)
{  
  initialise(parameter_filename);
  cerr<<parameter_info() << endl;
}

FBP2DReconstruction::FBP2DReconstruction()
{
  set_defaults();
}

FBP2DReconstruction::
FBP2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, 
		    const double alpha_ramp,
		    const double fc_ramp,
		    const int pad_in_s,
		    const int num_segments_to_combine
)
:		    alpha_ramp(alpha_ramp),
 		    fc_ramp(fc_ramp),
		    pad_in_s(pad_in_s),
		    num_segments_to_combine(num_segments_to_combine)
{
  // TODO bad to have this repeated from set_defaults()
  back_projector_sptr =
    new BackProjectorByBinUsingInterpolation(
					     /*use_piecewise_linear_interpolation = */true, 
					     /*use_exact_Jacobian = */ false);
  proj_data_ptr = proj_data_ptr_v;
  // have to check here because we're not parsing
  if (post_processing_only_FBP2D_parameters() == true)
    error("FBP2D: Wrong parameter values. Aborting\n");
}

Succeeded FBP2DReconstruction::
reconstruct()
{
  return Reconstruction::reconstruct();
}

Succeeded 
FBP2DReconstruction::
reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{

  if (dynamic_cast<const ProjDataInfoCylindricalArcCorr*>
       (proj_data_ptr->get_proj_data_info_ptr()) == 0)
  {
    warning("Projection data has to be arc-corrected for FBP2D\n");
    return Succeeded::no;
  }

  const ProjDataInfoCylindricalArcCorr& proj_data_info_cyl =
    dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
    (*proj_data_ptr->get_proj_data_info_ptr());

  // perform SSRB
  if (num_segments_to_combine>1)
    {  
      //  full_log << "SSRB combining " << num_segments_to_combine 
      //           << " segments in input file to a new segment 0\n" << endl; 

      shared_ptr<ProjData> proj_data_to_FBP_ptr = 
        new ProjDataInMemory (SSRB(proj_data_info_cyl, 
				   num_segments_to_combine,
				   1, 0,
				   (num_segments_to_combine-1)/2 ));
      SSRB(*proj_data_to_FBP_ptr, *proj_data_ptr);
      proj_data_ptr = proj_data_to_FBP_ptr;
    }
  else
    {
      // just use the proj_data_ptr we have already
    }


  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);

  assert(fabs(proj_data_ptr->get_proj_data_info_ptr()->get_tantheta(Bin(0,0,0,0)) ) < 1.E-4);

  // set projector to be used for the calculations
  back_projector_sptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(), 
			      density_ptr);


  // set ramp filter with appropriate sizes
  const int fft_size = (int) pow(2.,(int) ceil(log((double)(pad_in_s + 1)* proj_data_ptr->get_num_tangential_poss()) / log(2.)));
  
  RampFilter filter(proj_data_info_cyl.get_tangential_sampling(), 
			 fft_size, 
			 float(alpha_ramp), float(fc_ramp));   


  density_ptr->fill(0);
  
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    back_projector_sptr->get_symmetries_used()->clone();
  for (int view_num=proj_data_ptr->get_min_view_num(); view_num <= proj_data_ptr->get_max_view_num(); ++view_num) 
  {         
    const ViewSegmentNumbers vs_num(view_num, 0);
    if (!symmetries_sptr->is_basic(vs_num))
      continue;

    RelatedViewgrams<float> viewgrams = 
      proj_data_ptr->get_related_viewgrams(vs_num, symmetries_sptr);   

    // now filter
    for (RelatedViewgrams<float>::iterator viewgram_iter = viewgrams.begin();
         viewgram_iter != viewgrams.end();
         ++viewgram_iter)
    {
#ifdef NRFFT
      filter.apply(*viewgram_iter);
#else
      std::for_each(viewgram_iter->begin(), viewgram_iter->end(), 
		    filter);
#endif
    }

  if(display_level>1) 
    display( viewgrams,viewgrams.find_max(),"Ramp filter");

    //  and backproject
    back_projector_sptr->back_project(*density_ptr, viewgrams);
  } 
 
  // Normalise the image
  // KT & Darren Hogg 17/05/2000 finally found the scale factor!
  // TODO remove magic, is a scale factor in the backprojector 
  const float magic_number=2*proj_data_info_cyl.get_ring_radius()*proj_data_info_cyl.get_num_views()/proj_data_info_cyl.get_ring_spacing();
#ifdef NEWSCALE
  // added binsize etc here to get units ok
  // only do this when the forward projector units are appropriate
  image *= magic_number / proj_data_ptr->get_num_views() *
    proj_data_info_cyl.get_bin_size()/
    (image.get_voxel_size().x()*image.get_voxel_size().y());
#else
  image *= magic_number / proj_data_ptr->get_num_views();
#endif

  if (display_level>0)
    display(image, image.find_max(), "FBP image");

  return Succeeded::yes;
}

 

END_NAMESPACE_STIR
