/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2010, Hammersmith Imanet Ltd
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
  \ingroup recontest
  \brief Testing program for forward projection

  \author Kris Thielemans
  \author PARAPET project

  This program allows forward projection of a few segments/views
  only, or of the full data set. 

  \par Usage:
  \verbatim
  fwdtest output-filename template_proj_data_file [image_to_forward_project [forwardprojector-parfile ]]]\n"
  \endverbatim
  The template_proj_data_file will be used to get the scanner, mashing etc. details
  (its data will \e not be used, nor will it be overwritten).
  If some of these parameters are not given, some questions are asked.        

  \par Example parameter file for specifying the forward projector
  \verbatim
  Forward Projector parameters:=
    type := Matrix
      Forward projector Using Matrix Parameters :=
        Matrix type := Ray Tracing
         Ray tracing matrix parameters :=
         End Ray tracing matrix parameters :=
        End Forward Projector Using Matrix Parameters :=
  End:=
  \endverbatim
*/

#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/display.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"
#include "stir/ExamInfo.h"
// for ask_filename...
#include "stir/utilities.h"
#include "stir/round.h"
#include "stir/IndexRange3D.h"
#include "stir/RelatedViewgrams.h"
#include "stir/SegmentByView.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include <fstream>


using std::cerr;
using std::cout;
using std::endl;

USING_NAMESPACE_STIR
//USING_NAMESPACE_STD


/******************* Declarations local functions *******************/

static void 
do_segments(const VoxelsOnCartesianGrid<float>& image, ProjData& s3d,
	    const int start_timing_pos_num, const int end_timing_pos_num,
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num,
	    ForwardProjectorByBin&,
            const bool disp);
static void 
fill_cuboid(VoxelsOnCartesianGrid<float>& image);
static void 
fill_cylinder(VoxelsOnCartesianGrid<float>& image);



/*************************** main ***********************************/
int 
main(int argc, char *argv[])
{

  if(argc<3 || argc>5) 
  {
    std::cerr <<"Usage:\n"
	      << argv[0] << " \\\n"
	      << "    output-filename template_proj_data_file [image_to_forward_project [forwardprojector-parfile ]]]\n"
        <<"The template_proj_data_file will be used to get the scanner, mashing etc. details.\n";
    exit(EXIT_FAILURE);
  }
  
  const std::string output_file_name = argv[1];

  shared_ptr<ProjDataInfo> new_data_info_ptr;
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);  
  if(argc>=3)
  {
    shared_ptr<ProjData> proj_data_sptr = 
      ProjData::read_from_file(argv[2]);
    exam_info_sptr = proj_data_sptr->get_exam_info_sptr();
    new_data_info_ptr= proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone();
  }
  else
  {
    new_data_info_ptr.reset(ProjDataInfo::ask_parameters());
  }
  int limit_segments=
    ask_num("Maximum absolute segment number to process: ", 0, 
    new_data_info_ptr->get_max_segment_num(), 
    new_data_info_ptr->get_max_segment_num() );

  new_data_info_ptr->reduce_segment_range(-limit_segments, limit_segments);

  shared_ptr<ProjData> proj_data_ptr(new ProjDataInterfile(exam_info_sptr, new_data_info_ptr, output_file_name));
  cerr << "Output will be written to " << output_file_name 
       << " and its Interfile header\n";

  
  int dispstart = 0;
  int save = 0;
  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr;   
  VoxelsOnCartesianGrid<float> * vox_image_ptr = 0;

  if (argc<4)
    {
      switch (int choice = ask_num("Start image is cuboid (1) or cylinder (2) or on file (3)",1,3,2))
	{
	case 1: 
	case 2:
	  {
	    dispstart = 
	      ask_num("Display start image ? no (0), yes (1)", 0,1,0);

	    save = 
	      ask_num("Save  start images ? no (0), yes (1)", 0,1,0);   
	    const float zoom = ask_num("Zoom factor (>1 means smaller voxels)",0.F,10.F,1.F);
	    int xy_size = static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom);
	    xy_size = ask_num("Number of x,y pixels",3,xy_size*2,xy_size);
	    const float zoom_z = ask_num("Zoom factor in z", 0.F, 10.F, 1.F);
	    int z_size = 
	      stir::round((2*proj_data_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()-1)*zoom_z);
	    z_size = ask_num("Number of z pixels",1,1000,z_size);
	    vox_image_ptr =
	      new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()),
					       zoom,
					       CartesianCoordinate3D<float>(0,0,0),
					       Coordinate3D<int>(z_size,xy_size,xy_size));
	    image_sptr.reset(vox_image_ptr);
	    vox_image_ptr->set_grid_spacing(
					    vox_image_ptr->get_grid_spacing()/
					    CartesianCoordinate3D<float>(zoom_z,1.F,1.F));
	    if (choice==1)
	      fill_cuboid(*vox_image_ptr);
	    else
	      fill_cylinder(*vox_image_ptr);
	    break;
	  }
	case 3:
	  {
	    char filename[max_filename_length];
      
	    ask_filename_with_extension(filename, "Input file name ?", ".hv");
      
	    image_sptr =
	      read_from_file<DiscretisedDensity<3,float> >(filename);
	    vox_image_ptr = dynamic_cast<VoxelsOnCartesianGrid<float> *> (image_sptr.get());
      
	    break;
	  }
	}
    }
  else
    {
      image_sptr =
        read_from_file<DiscretisedDensity<3,float> >(argv[3]);
      vox_image_ptr = dynamic_cast<VoxelsOnCartesianGrid<float> *> (image_sptr.get());
    }

  const float z_origin = 
    ask_num("Shift z-origin (in pixels)", 
             -vox_image_ptr->get_length()/2,
             vox_image_ptr->get_length()/2,
             0) *vox_image_ptr->get_voxel_size().z();
  
  vox_image_ptr->set_origin(make_coordinate(z_origin,0.F,0.F) + vox_image_ptr->get_origin());
  // use shared_ptr such that it cleans up automatically
  shared_ptr<ForwardProjectorByBin> forw_projector_ptr;

  if (argc>=5)
    {
      KeyParser parser;
      parser.add_start_key("Forward Projector parameters");
      parser.add_parsing_key("type", &forw_projector_ptr);
      parser.add_stop_key("END"); 
      parser.parse(argv[4]);
    }

  while (is_null_ptr(forw_projector_ptr))
    {
      forw_projector_ptr.reset(ForwardProjectorByBin::ask_type_and_parameters());
    }

  forw_projector_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone(),
			     image_sptr);
  cerr << forw_projector_ptr->parameter_info();

  if (dispstart)
    {
      cerr << "Displaying start image";
      display(*image_sptr, image_sptr->find_max());
    }
  

  if (save)
  {
    cerr << "Saving start image to 'test_image'" << endl;
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
      write_to_file("test_image", *image_sptr);
  }
    
  std::list<ViewSegmentNumbers> already_processed;

  if (ask("Do full forward projection ?", true))
  {    

    CPUTimer timer;
    timer.reset();
    timer.start();

    
    do_segments(*vox_image_ptr, *proj_data_ptr,
	  proj_data_ptr->get_min_tof_pos_num(), proj_data_ptr->get_max_tof_pos_num(),
      proj_data_ptr->get_min_segment_num(), proj_data_ptr->get_max_segment_num(), 
      proj_data_ptr->get_min_view_num(), 
      proj_data_ptr->get_max_view_num(),
      proj_data_ptr->get_min_tangential_pos_num(), 
      proj_data_ptr->get_max_tangential_pos_num(),
      *forw_projector_ptr,
      false);
    
    timer.stop();
    cerr << timer.value() << " s CPU time"<<endl;
    

  }
  else
  { 
    const bool disp = 
      ask("Display projected related viewgrams ? ", false);
    
    // first set all data to 0
    cerr << "Filling output file with 0\n";
	for (int timing_pos_num = proj_data_ptr->get_min_tof_pos_num();
		timing_pos_num <= proj_data_ptr->get_max_tof_pos_num();
		++timing_pos_num)
    for (int segment_num = proj_data_ptr->get_min_segment_num(); 
         segment_num <= proj_data_ptr->get_max_segment_num(); 
         ++segment_num)
    {
      const SegmentByView<float> segment = 
        proj_data_ptr->get_empty_segment_by_view(segment_num, false,timing_pos_num);
      if (!(proj_data_ptr->set_segment(segment) == Succeeded::yes))
        warning("Error set_segment %d of timing position index %d\n", segment_num,timing_pos_num);            
    }
    do
    {
      CPUTimer timer;
      timer.reset();
      timer.start();
      
	  const int timing_pos_num =
		  ask_num("Timing position index to forward project",
			  proj_data_ptr->get_min_tof_pos_num(),
			  proj_data_ptr->get_max_tof_pos_num(),
			  0);
      const int segment_num = 
        ask_num("Segment number to forward project (related segments will be done as well)",
                proj_data_ptr->get_min_segment_num(),
                proj_data_ptr->get_max_segment_num(), 
                0);
      const int min_view = proj_data_ptr->get_min_view_num();
      const int max_view = proj_data_ptr->get_max_view_num();

      const int start_view = 
        ask_num("Start view  (related views will be done as well)", min_view, max_view, min_view);
      const int end_view = 
        ask_num("End   view  (related views will be done as well)", start_view, max_view, max_view);

      const int min_tangential_pos_num = proj_data_ptr->get_min_tangential_pos_num();
      const int max_tangential_pos_num = proj_data_ptr->get_max_tangential_pos_num();

      const int start_tangential_pos_num = 
        ask_num("Start tangential_pos_num", min_tangential_pos_num, max_tangential_pos_num, min_tangential_pos_num);
      const int end_tangential_pos_num = 
        ask_num("End   tangential_pos_num ", start_tangential_pos_num, max_tangential_pos_num, max_tangential_pos_num);
      
      do_segments(*vox_image_ptr,*proj_data_ptr, 
		      timing_pos_num, timing_pos_num,
	          segment_num,segment_num, 
	          start_view, end_view,
                  start_tangential_pos_num, end_tangential_pos_num,
	          *forw_projector_ptr, disp);
      

      timer.stop();
      cerr << timer.value() << " s CPU time"<<endl;

    }
    while (ask("One more ? ", true));
  }

  return EXIT_SUCCESS;
  
}

/******************* Implementation local functions *******************/
void
do_segments(const VoxelsOnCartesianGrid<float>& image,
	ProjData& proj_data,
	const int start_timing_pos_num, const int end_timing_pos_num,
	const int start_segment_num, const int end_segment_num,
	const int start_view, const int end_view,
	const int start_tangential_pos_num, const int end_tangential_pos_num,
	ForwardProjectorByBin& forw_projector,
	const bool disp)
{
	shared_ptr<DataSymmetriesForViewSegmentNumbers>
		symmetries_sptr(forw_projector.get_symmetries_used()->clone());

	std::list<ViewSegmentNumbers> already_processed;
	for (int timing_pos_num = start_timing_pos_num; timing_pos_num <= end_timing_pos_num; ++timing_pos_num)
	{
		already_processed.clear();
		for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
			for (int view = start_view; view <= end_view; view++)
			{
				ViewSegmentNumbers vs(view, segment_num);
				symmetries_sptr->find_basic_view_segment_numbers(vs);
				if (find(already_processed.begin(), already_processed.end(), vs)
					!= already_processed.end())
					continue;

				already_processed.push_back(vs);

				cerr << "Processing view " << vs.view_num()
					<< " of segment " << vs.segment_num()
					<< " of timing position index " << timing_pos_num
					<< endl;

				RelatedViewgrams<float> viewgrams =
					proj_data.get_empty_related_viewgrams(vs, symmetries_sptr, false,timing_pos_num);
				forw_projector.forward_project(viewgrams, image,
					viewgrams.get_min_axial_pos_num(),
					viewgrams.get_max_axial_pos_num(),
					start_tangential_pos_num, end_tangential_pos_num);
				if (disp)
					display(viewgrams, viewgrams.find_max());
				if (!(proj_data.set_related_viewgrams(viewgrams) == Succeeded::yes))
					error("Error set_related_viewgrams\n");
			}
	}
}



void fill_cuboid(VoxelsOnCartesianGrid<float>& image)
{
  const float voxel_value = ask_num("Voxel value",-10E10F, 10E10F,1.F);
  const int xs = ask_num("Start X coordinate", 
			 image.get_min_x(), image.get_max_x(), 
			 (image.get_min_x()+ image.get_max_x())/2);
  const int ys = ask_num("Start Y coordinate", 
			 image.get_min_y(), image.get_max_y(), 
			 (image.get_min_y()+ image.get_max_y())/2);
  const int zs = ask_num("Start Z coordinate", 
			 image.get_min_z(), image.get_max_z(), 
			 (image.get_min_z()+ image.get_max_z())/2);
  
  const int xe = ask_num("End X coordinate", xs, image.get_max_x(), xs);
  const int ye = ask_num("End Y coordinate", ys, image.get_max_y(), ys);
  const int ze = ask_num("End Z coordinate", zs, image.get_max_z(), zs);
  
  
  cerr << "Start coordinate: (x,y,z) = (" 
       << xs << ", " << ys << ", " << zs  
       << ")" << endl;
  cerr << "End   coordinate: (x,y,z) = (" 
       << xe << ", " << ye << ", " << ze  
       << ")" << endl;	
  
  image.fill(0);
  for (int z=zs; z<=ze; z++)
    for (int y=ys; y <= ye; y++)
      for (int x=xs; x<= xe; x++)
	image[z][y][x] = voxel_value; 
}

void fill_cylinder(VoxelsOnCartesianGrid<float>& image)
{
  const float voxel_value = ask_num("Voxel value",-10E10F, 10E10F,1.F);
 
  const double xc = 
    ask_num("Centre X coordinate", 
	    (double)image.get_min_x(), (double)image.get_max_x(), 
	    (image.get_min_x()+ image.get_max_x())/2.);
  
  const double yc = 
    ask_num("Centre Y coordinate", 
	    (double)image.get_min_y(), (double)image.get_max_y(), 
	    (image.get_min_y()+ image.get_max_y())/2.);
  
  const float zc = 
    ask_num("Centre Z coordinate",     
	    (float)image.get_min_z(), (float)image.get_max_z(), 
	    (image.get_min_z()+ image.get_max_z())/2.F);

  
  const double Rcyl = 
    ask_num("Radius (pixels)", 
	    .5, (image.get_max_x()- image.get_min_x())/2.,
	    (image.get_max_x()- image.get_min_x())/4.);
  
  // Max length is num_planes+1 because of edges of voxels
  const float Lcyl = 
    ask_num("Length (planes)", 1.F, (image.get_max_z()- image.get_min_z())+1.F,
	    (image.get_max_z()- image.get_min_z())+1.F);
  
  
  cerr << "Centre coordinate: (x,y,z) = (" 
       << xc << ", " << yc << ", " << zc  
       << ")" << endl;
  cerr << "Radius = " << Rcyl << ", Length = " << Lcyl << endl; 
  
  const int num_samples = 
    ask_num("With how many points (in x,y direction) do I sample each voxel ?",
	    1,100,5);
  
  Array<2,float> plane = image[0];
  
  for (int y=image.get_min_y(); y<=image.get_max_y(); y++)
    for (int x=image.get_min_x(); x<=image.get_max_x(); x++)
      {
	double value = 0;
      
	for (double ysmall=-(num_samples-1.)/num_samples/2.; 
	     ysmall < 0.5; 
	     ysmall+= 1./num_samples)
	  {
	    const double ytry = y-ysmall-yc;
	
	    for (double xsmall=-(num_samples-1.)/num_samples/2.; 
		 xsmall < 0.5; 
		 xsmall+= 1./num_samples)
	      {
		const double xtry = x-xsmall-xc;
	  
		if (xtry*xtry + ytry*ytry <= Rcyl*Rcyl)
		  value++;
	      }
	  }
	// update plane with normalised value (independent of num_samples)
	plane[y][x] = static_cast<float>(voxel_value*value/(num_samples*num_samples)); 
      }
    
  for (int z=image.get_min_z(); z<=image.get_max_z(); z++)
    {
      // use 2. to make both args of min() and max() double
    float zfactor = (std::min(z+.5F, zc+Lcyl/2.F) - std::max(z-.5F, zc-Lcyl/2.F));
      if (zfactor<0) zfactor = 0;
      image[z] = plane;
      image[z] *= zfactor;
    }
    
}

