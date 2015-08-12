/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012-01-09, Hammersmith Imanet Ltd
    Copyright (C) 2013 University College London

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
  \ingroup FBP2D
  \brief Implementation of class stir::FBP2DReconstruction

  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project
*/

#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ArcCorrection.h"
#include "stir/analytic/FBP2D/RampFilter.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
// #include "stir/ProjDataInterfile.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
#include "stir/info.h"

#ifdef STIR_OPENMP
#include <omp.h>
#endif

using std::cerr;
using std::endl;

START_NAMESPACE_STIR


void 
FBP2DReconstruction::
set_defaults()
{
  base_type::set_defaults();

  alpha_ramp = 1;
  fc_ramp = 0.5;
  pad_in_s=1;
  display_level=0; // no display
  num_segments_to_combine = -1;
  back_projector_sptr.reset(new BackProjectorByBinUsingInterpolation(
								     /*use_piecewise_linear_interpolation = */true, 
								     /*use_exact_Jacobian = */ false));
 
}

void 
FBP2DReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

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
   
  base_type::ask_parameters();

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
  if (base_type::post_processing())
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
        num_segments_to_combine = 1; //cannot SSRB non-cylindrical data yet
      else
	{
	  if (proj_data_info_cyl_ptr->get_min_ring_difference(0) != 
	      proj_data_info_cyl_ptr->get_max_ring_difference(0)
	      ||
	      proj_data_info_cyl_ptr->get_num_segments()==1)
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

std::string FBP2DReconstruction::method_info() const
{
  return "FBP2D";
}

FBP2DReconstruction::
FBP2DReconstruction(const std::string& parameter_filename)
{  
  initialise(parameter_filename);
  info(boost::format("%1%") % parameter_info());
}

FBP2DReconstruction::FBP2DReconstruction()
{
  set_defaults();
}

FBP2DReconstruction::
FBP2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, 
		    const double alpha_ramp_v,
		    const double fc_ramp_v,
		    const int pad_in_s_v,
		    const int num_segments_to_combine_v
)
{
  set_defaults();

  alpha_ramp = alpha_ramp_v;
  fc_ramp = fc_ramp_v;
  pad_in_s = pad_in_s_v;
  num_segments_to_combine = num_segments_to_combine_v;
  proj_data_ptr = proj_data_ptr_v;
  // have to check here because we're not parsing
  if (post_processing_only_FBP2D_parameters() == true)
    error("FBP2D: Wrong parameter values. Aborting\n");
}

Succeeded 
FBP2DReconstruction::
actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & density_ptr)
{

  // perform SSRB
  if (num_segments_to_combine>1)
    {  
      const ProjDataInfoCylindrical& proj_data_info_cyl =
	dynamic_cast<const ProjDataInfoCylindrical&>
	(*proj_data_ptr->get_proj_data_info_ptr());

      //  full_log << "SSRB combining " << num_segments_to_combine 
      //           << " segments in input file to a new segment 0\n" << endl; 

      shared_ptr<ProjDataInfo> 
	ssrb_info_sptr(SSRB(proj_data_info_cyl, 
			    num_segments_to_combine,
			    1, 0,
			    (num_segments_to_combine-1)/2 ));
      shared_ptr<ProjData> 
	proj_data_to_FBP_ptr(new ProjDataInMemory (proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
      SSRB(*proj_data_to_FBP_ptr, *proj_data_ptr);
      proj_data_ptr = proj_data_to_FBP_ptr;
    }
  else
    {
      // just use the proj_data_ptr we have already
    }

  // check if segment 0 has direct sinograms
  {
    const float tan_theta = proj_data_ptr->get_proj_data_info_ptr()->get_tantheta(Bin(0,0,0,0));
    if(fabs(tan_theta ) > 1.E-4)
      {
	warning("FBP2D: segment 0 has non-zero tan(theta) %g", tan_theta);
	return Succeeded::no;
      }
  }

  float tangential_sampling;
  // TODO make next type shared_ptr<ProjDataInfoCylindricalArcCorr> once we moved to boost::shared_ptr
  // will enable us to get rid of a few of the ugly lines related to tangential_sampling below
  shared_ptr<ProjDataInfo> arc_corrected_proj_data_info_sptr;

  // arc-correction if necessary
  ArcCorrection arc_correction;
  bool do_arc_correction = false;
  if (dynamic_cast<const ProjDataInfoCylindricalArcCorr*>
      (proj_data_ptr->get_proj_data_info_ptr()) != 0)
    {
      // it's already arc-corrected
      arc_corrected_proj_data_info_sptr =
	proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone();
      tangential_sampling =
	dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
	(*proj_data_ptr->get_proj_data_info_ptr()).get_tangential_sampling();  
    }
  else
    {
      // TODO arc-correct to voxel_size
      if (arc_correction.set_up(proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone()) ==
	  Succeeded::no)
	return Succeeded::no;
      do_arc_correction = true;
      // TODO full_log
      warning("FBP2D will arc-correct data first");
      arc_corrected_proj_data_info_sptr =
	arc_correction.get_arc_corrected_proj_data_info_sptr();
      tangential_sampling =
	arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();  
    }
  //ProjDataInterfile ramp_filtered_proj_data(arc_corrected_proj_data_info_sptr,"ramp_filtered");

  VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);


  // set projector to be used for the calculations
  back_projector_sptr->set_up(arc_corrected_proj_data_info_sptr, 
			      density_ptr);


  // set ramp filter with appropriate sizes
  const int fft_size = 
    round(pow(2., ceil(log((double)(pad_in_s + 1)* arc_corrected_proj_data_info_sptr->get_num_tangential_poss()) / log(2.))));
  
  RampFilter filter(tangential_sampling,
			 fft_size, 
			 float(alpha_ramp), float(fc_ramp));   


  density_ptr->fill(0);
  
  shared_ptr<DataSymmetriesForViewSegmentNumbers> 
    symmetries_sptr(back_projector_sptr->get_symmetries_used()->clone());
    
#ifdef STIR_OPENMP
  if (getenv("OMP_NUM_THREADS")==NULL) 
    {
      omp_set_num_threads(omp_get_num_procs());
      if (omp_get_num_procs()==1) 
	warning("Using OpenMP with #processors=1 produces parallel overhead. You should compile without using USE_OPENMP=TRUE.");
      info(boost::format("Using OpenMP-version of FBP2D with thread-count = processor-count (=%1%).") % omp_get_num_procs());
    }
  else 
    {
      info(boost::format("Using OpenMP-version of FBP2D with %1% threads on %2% processors.") % getenv("OMP_NUM_THREADS") % omp_get_num_procs());
      if (atoi(getenv("OMP_NUM_THREADS"))==1) 
	warning("Using OpenMP with OMP_NUM_THREADS=1 produces parallel overhead. Use more threads or compile without using USE_OPENMP=TRUE.");
    }
  info("Define number of threads by setting OMP_NUM_THREADS environment variable, i.e. \"export OMP_NUM_THREADS=<num_threads>\"");
  shared_ptr<DiscretisedDensity<3,float> > empty_density_ptr(density_ptr->clone());
#endif

#ifdef STIR_OPENMP
#pragma omp parallel for shared(empty_density_ptr) schedule(dynamic)
#endif
  for (int view_num=proj_data_ptr->get_min_view_num(); view_num <= proj_data_ptr->get_max_view_num(); ++view_num) 
  {         
    const ViewSegmentNumbers vs_num(view_num, 0);
    
#ifndef NDEBUG
#ifdef STIR_OPENMP
    info(boost::format("Thread %1% calculating view_num: %2%") % omp_get_thread_num() % view_num);
#endif 
#endif
    
    if (!symmetries_sptr->is_basic(vs_num))
      continue;

    RelatedViewgrams<float> viewgrams;
#ifdef STIR_OPENMP
#pragma omp critical(FBP2D_get_viewgrams)
#endif
    {
      viewgrams =
	proj_data_ptr->get_related_viewgrams(vs_num, symmetries_sptr);   
    }

    if (do_arc_correction)
      viewgrams =
	arc_correction.do_arc_correction(viewgrams);

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
    // ramp_filtered_proj_data.set_related_viewgrams(viewgrams);

  if(display_level>1) 
    display( viewgrams,viewgrams.find_max(),"Ramp filter");

#ifdef STIR_OPENMP 
  //clone density_ptr and backproject    
  shared_ptr<DiscretisedDensity<3,float> > omp_density_ptr(empty_density_ptr->clone());
           
    back_projector_sptr->back_project(*omp_density_ptr, viewgrams);
#pragma omp critical(FBP2D_REDUCTION)
    {	//reduction
      
      DiscretisedDensity<3,float>::full_iterator density_iter = density_ptr->begin_all();
      DiscretisedDensity<3,float>::full_iterator density_end = density_ptr->end_all();
      DiscretisedDensity<3,float>::full_iterator omp_density_iter = omp_density_ptr->begin_all();
      
      while (density_iter!= density_end)
	{
	  *density_iter += (*omp_density_iter);
	  ++density_iter;
	  ++omp_density_iter;
	}
    }
#else
    //  and backproject
    back_projector_sptr->back_project(*density_ptr, viewgrams);
#endif
  } 
 
  // Normalise the image
  const ProjDataInfoCylindrical& proj_data_info_cyl =
    dynamic_cast<const ProjDataInfoCylindrical&>
    (*proj_data_ptr->get_proj_data_info_ptr());

  float magic_number = 1.F;
  if (dynamic_cast<BackProjectorByBinUsingInterpolation const *>(back_projector_sptr.get()) != 0)
    {
      // KT & Darren Hogg 17/05/2000 finally found the scale factor!
      // TODO remove magic, is a scale factor in the backprojector 
      magic_number=2*proj_data_info_cyl.get_ring_radius()*proj_data_info_cyl.get_num_views()/proj_data_info_cyl.get_ring_spacing();
    }
  else
    {
      if (proj_data_info_cyl.get_min_ring_difference(0)!=
	  proj_data_info_cyl.get_max_ring_difference(0))
	{
	  magic_number=.5F;
	}
    }
#ifdef NEWSCALE
  // added binsize etc here to get units ok
  // only do this when the forward projector units are appropriate
  image *= magic_number / proj_data_ptr->get_num_views() *
    tangential_sampling/
    (image.get_voxel_size().x()*image.get_voxel_size().y());
#else
  image *= magic_number / proj_data_ptr->get_num_views();
#endif

  if (display_level>0)
    display(image, image.find_max(), "FBP image");

  return Succeeded::yes;
}

 

END_NAMESPACE_STIR
