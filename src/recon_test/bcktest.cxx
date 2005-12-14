//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup recontest

  \brief Test program for back projection

  This program is mainly intended for testing backprojectors, but it can be
  used (somewhat awkwardly) to backproject sinograms onto images.

  \par Usage

  \verbatim
  bcktest [output-filename [proj_data_file \
       [template-image [backprojector-parfile ]]]]
  \endverbatim
  If some command line parameter is not given, the program will ask 
  the user interactively.

  The format of the parameter file to specifiy the backproejctor is 
  as follows:
  \verbatim
      Back Projector parameters:=
      type:= some_type
      END:=
  \endverbatim
  \see stir::BackprojectorByBin for derived classes and their parameters.

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  
  $Date$
  $Revision$
*/

#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/display.h"
#include "stir/KeyParser.h"
#include "stir/stream.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfo.h"
// for ask_filename...
#include "stir/utilities.h"
#include "stir/IndexRange3D.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

#include <fstream>
#include <list>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::iostream;
using std::endl;
using std::list;
using std::find;
using std::cerr;
using std::endl;
#endif




START_NAMESPACE_STIR

void
do_segments(DiscretisedDensity<3,float>& image, 
            ProjData& proj_data_org,
	    const int start_segment_num, const int end_segment_num,
	    const int start_axial_pos_num, const int end_axial_pos_num,
	    const int start_tang_pos_num,const int end_tang_pos_num,
	    const int start_view, const int end_view,
	    BackProjectorByBin* back_projector_ptr,
	    bool fill_with_1)
{
  
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    back_projector_ptr->get_symmetries_used()->clone();
  
  
  list<ViewSegmentNumbers> already_processed;
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    for (int view= start_view; view<=end_view; view++)
  { 
    ViewSegmentNumbers vs(view, segment_num);
    symmetries_sptr->find_basic_view_segment_numbers(vs);
    if (find(already_processed.begin(), already_processed.end(), vs)
        != already_processed.end())
      continue;

    already_processed.push_back(vs);
    
    cerr << "Processing view " << vs.view_num()
      << " of segment " <<vs.segment_num()
      << endl;
    
    if(fill_with_1 )
    {
      RelatedViewgrams<float> viewgrams_empty= 
	proj_data_org.get_empty_related_viewgrams(vs, symmetries_sptr);
	//proj_data_org.get_empty_related_viewgrams(vs.view_num(),vs.segment_num(), symmetries_sptr);
      
      RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams_empty.begin();
      while(r_viewgrams_iter!=viewgrams_empty.end())
      {
	Viewgram<float>&  single_viewgram = *r_viewgrams_iter;
	if (start_view <= single_viewgram.get_view_num() && 
	  single_viewgram.get_view_num() <= end_view &&
	  single_viewgram.get_segment_num() >= start_segment_num &&
	  single_viewgram.get_segment_num() <= end_segment_num)
	{
	  single_viewgram.fill(1.F);
	}
	r_viewgrams_iter++;	  
      } 
      
      back_projector_ptr->back_project(image,viewgrams_empty,
	max(start_axial_pos_num, viewgrams_empty.get_min_axial_pos_num()), 
	min(end_axial_pos_num, viewgrams_empty.get_max_axial_pos_num()),
	start_tang_pos_num, end_tang_pos_num);
    }
    else
    {
      RelatedViewgrams<float> viewgrams = 
	proj_data_org.get_related_viewgrams(vs,
	//proj_data_org.get_related_viewgrams(vs.view_num(),vs.segment_num(),
	symmetries_sptr);
      RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams.begin();
      
      while(r_viewgrams_iter!=viewgrams.end())
      {
	Viewgram<float>&  single_viewgram = *r_viewgrams_iter;
	{	  
	  if (start_view <= single_viewgram.get_view_num() && 
	    single_viewgram.get_view_num() <= end_view &&
	  single_viewgram.get_segment_num() >= start_segment_num &&
	  single_viewgram.get_segment_num() <= end_segment_num)
	  {
	    // ok
	  }
	  else
	  { 
	    // set to 0 to prevent it being backprojected
	    single_viewgram.fill(0);
	  }
	}
	++r_viewgrams_iter;
      }
	
      back_projector_ptr->back_project(image,viewgrams,
	  max(start_axial_pos_num, viewgrams.get_min_axial_pos_num()), 
	  min(end_axial_pos_num, viewgrams.get_max_axial_pos_num()),
	  start_tang_pos_num, end_tang_pos_num);      
    } // fill
  } // for view_num, segment_num    
    
}

END_NAMESPACE_STIR



USING_NAMESPACE_STIR
int
main(int argc, char **argv)
{  
  if (argc==1 || argc>7)
  {
      cerr <<"Usage: " << argv[0] << " \\\n"
	   << "\t[output-filename [proj_data_file [template-image [backprojector-parfile ]]]]\n";
      exit(EXIT_FAILURE);
    }
  const string output_filename=
    argc>1? argv[1] : ask_string("Output filename");

  shared_ptr<ProjData> proj_data_ptr;
  bool fill;

  if (argc>2)
    { 
      proj_data_ptr = ProjData::read_from_file(argv[2]); 
      fill = ask("Do you want to backproject all 1s (Y) or the data (N) ?", true);
    }
  else
    {
      shared_ptr<ProjDataInfo> data_info= ProjDataInfo::ask_parameters();
      // create an empty ProjDataFromStream object
      // such that we don't have to differentiate between code later on
      proj_data_ptr = 
	new ProjDataFromStream (data_info,static_cast<iostream *>(NULL));
      fill = true;
    }
  shared_ptr<BackProjectorByBin> back_projector_ptr;

  const bool disp = ask("Display images ?", false);
  
  const bool save = ask("Save  images ?", true);
    
  const bool save_profiles = ask("Save  horizontal profiles ?", false);

  // note: first check 4th parameter for historical reasons 
  // (could be switched with 3rd without problems)
  if (argc>4)
    {
      KeyParser parser;
      parser.add_start_key("Back Projector parameters");
      parser.add_parsing_key("type", &back_projector_ptr);
      parser.add_stop_key("END"); 
      parser.parse(argv[4]);
    }

  const ProjDataInfo * proj_data_info_ptr =
    proj_data_ptr->get_proj_data_info_ptr();
 
  shared_ptr<DiscretisedDensity<3,float> > image_sptr;

  if (argc>3)
    {
      image_sptr = DiscretisedDensity<3,float>::read_from_file(argv[3]);
    }
  else
    {
      const float zoom = ask_num("Zoom factor (>1 means smaller voxels)",0.F,100.F,1.F);
      int xy_size = static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom);
      xy_size = ask_num("Number of x,y pixels",3,xy_size*2,xy_size);
      int z_size = 2*proj_data_info_ptr->get_scanner_ptr()->get_num_rings()-1;
      z_size = ask_num("Number of z pixels",1,1000,z_size);
      VoxelsOnCartesianGrid<float> * vox_image_ptr =
	new VoxelsOnCartesianGrid<float>(*proj_data_info_ptr,
					 zoom,
					 Coordinate3D<float>(0,0,0),
					 Coordinate3D<int>(z_size,xy_size,xy_size));
      const float z_origin = 
	ask_num("Shift z-origin (in pixels)", 
		-vox_image_ptr->get_length()/2,
		vox_image_ptr->get_length()/2,
		0)
	*vox_image_ptr->get_voxel_size().z();
      vox_image_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));

      image_sptr = vox_image_ptr;
    }

  while (is_null_ptr(back_projector_ptr))
    {
      back_projector_ptr =
	BackProjectorByBin::ask_type_and_parameters();
    }

  back_projector_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
			     image_sptr);
 
  do
  {    
    
    int min_segment_num = ask_num("Minimum segment number to backproject",
      proj_data_info_ptr->get_min_segment_num(), proj_data_info_ptr->get_max_segment_num(), 0);
    int max_segment_num = ask_num("Maximum segment number to backproject",
      min_segment_num,proj_data_info_ptr->get_max_segment_num(), 
      min_segment_num);
    
    // find max_axial_pos_num in the range of segments
    // TODO relies on axial_pos_num starting from 0
    assert(proj_data_info_ptr->get_min_axial_pos_num(0) == 0);
#if 1  
    int max_axial_pos_num;
    if (min_segment_num <= 0 && 0 <= max_segment_num)
    {
      // all axial_poss are addressed for segment 0
      max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(0);
    }
    else 
    {
      if (min_segment_num>0) // which implies max_segment_num>0
	max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(min_segment_num);      
      else // min_segment_num <= max_segment_num < 0      
	max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(max_segment_num);
    }
    
    const int start_axial_pos_num = 
      ask_num("Start axial_pos", 0, max_axial_pos_num, 0);
    const int end_axial_pos_num = 
      ask_num("End   axial_pos", start_axial_pos_num, max_axial_pos_num, max_axial_pos_num);
#else
    const int min_axial_pos_num = proj_data_info_ptr->get_min_axial_pos_num(segment_num);  
    const int max_axial_pos_num = proj_data_info_ptr->get_max_axial_pos_num(segment_num);  
    const int start_axial_pos_num = 
      ask_num("Start axial_pos", min_axial_pos_num, max_axial_pos_num, min_axial_pos_num);
    const int end_axial_pos_num = 
      ask_num("End   axial_pos", start_axial_pos_num, max_axial_pos_num, max_axial_pos_num);
#endif
    
    // TODO this message is symmetry specific
    const int nviews = proj_data_info_ptr->get_num_views();
    cerr << "Special views are at 0, "
      << nviews/4 <<", " << nviews/2 <<", " << nviews/4*3 << endl;
    
    int start_view = ask_num("Start view", 0, nviews-1, 0);
    int end_view = ask_num("End   view", 0, nviews-1, nviews-1);
    
    const int start_tang_pos_num = 
      ask_num("Start tang_pos", 
	      proj_data_info_ptr->get_min_tangential_pos_num(), 
	      proj_data_info_ptr->get_max_tangential_pos_num(),
	      proj_data_info_ptr->get_min_tangential_pos_num());
    const int end_tang_pos_num = 
      ask_num("End  tang_pos", 
	      start_tang_pos_num, 
	      proj_data_info_ptr->get_max_tangential_pos_num(),
	      proj_data_info_ptr->get_max_tangential_pos_num());
    //const int start_tang_pos_num = -end_tang_pos_num;

    if (!ask("Add this backprojection to image of previous run ?", false))
      image_sptr->fill(0);
    
    CPUTimer timer;
    timer.reset();
    back_projector_ptr->reset_timers();
    back_projector_ptr->start_timers();
    timer.start();

    do_segments(*image_sptr, 
      *proj_data_ptr, 
      min_segment_num, max_segment_num,
      start_axial_pos_num,end_axial_pos_num,
      start_tang_pos_num,end_tang_pos_num,
      start_view, end_view,
      back_projector_ptr.get(),
      fill);  
    
    timer.stop();
    cerr << timer.value() << " s total CPU time\n";
    cerr << "of which " << back_projector_ptr->get_CPU_timer_value()
	 << " s is reported by backprojector\n";
    cerr << "min and max in image " << image_sptr->find_min()
      << ", " << image_sptr->find_max() << endl;
    
    if (disp)
      display(*image_sptr, image_sptr->find_max());
    
    if (save)
    {
      DefaultOutputFileFormat output_format;
      cerr <<"  - Saving " << output_filename << endl;
      output_format.write_to_file(output_filename, *image_sptr);
      
    }

    if (save_profiles)
    {
      
      cerr << "Writing horizontal profiles to bcktest.prof" << endl;
      ofstream profile("bcktest.prof");
      if (!profile)
      { cerr << "Couldn't open " << "bcktest.prof"; }
      
      for (int z=image_sptr->get_min_index(); z<= image_sptr->get_max_index(); z++) 
	profile << (*image_sptr)[z][0] << '\n';
    }
    
  }
  while (ask("One more ?", true));

  return  EXIT_SUCCESS;
}


