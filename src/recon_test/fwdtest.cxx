//
// $Id$
//

/*!

  \file
  \ingroup recontest
  \brief Testing programme for forward projection

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$

  This programme allows forward projection of a few segments/views
  only, or of the full data set. 

  Usage:
  \verbatim
  fwdtest [proj_data_file]
  \endverbatim
  The proj_data_file will be used to get the scanner, mashing etc. details
  (its data will \e not be used, nor will it be overwritten).
  If no proj_data_file is given, some questions are asked to use 'standard'
  characteristics.        
*/

#include "recon_buildblock/ForwardProjectorByBin.h"
#include "display.h"
#include "interfile.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfo.h"
// for ask_filename...
#include "utilities.h"
#include "IndexRange3D.h"
#include "RelatedViewgrams.h"
#include "SegmentByView.h"
#include "VoxelsOnCartesianGrid.h"
#include <fstream>



USING_NAMESPACE_TOMO
USING_NAMESPACE_STD


/******************* Declarations local functions *******************/

static void 
do_segments(const VoxelsOnCartesianGrid<float>& image, ProjData& s3d,
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

  if(argc!=2) 
  {
    cerr<<"Usage: " << argv[0] << " [proj_data-file]\n"
        <<"The projdata-file will be used to get the scanner, mashing etc. details" 
	<< endl; 
  }
  

  ProjDataInfo* new_data_info_ptr;
  if(argc==2)
  {
    shared_ptr<ProjData> proj_data_ptr = 
      ProjData::read_from_file(argv[1]);
    new_data_info_ptr= proj_data_ptr->get_proj_data_info_ptr()->clone();
  }
  else
  {
    new_data_info_ptr= ProjDataInfo::ask_parameters();
  }
  int limit_segments=
    ask_num("Maximum absolute segment number to process: ", 0, 
    new_data_info_ptr->get_max_segment_num(), 
    new_data_info_ptr->get_max_segment_num() );

  new_data_info_ptr->reduce_segment_range(-limit_segments, limit_segments);

  const string output_file_name = "fwdtest_out.s";
  shared_ptr<iostream> sino_stream = new fstream (output_file_name.c_str(), ios::out|ios::binary);
  if (!sino_stream->good())
  {
    error("fwdtest: error opening file %s\n",output_file_name.c_str());
  }

  shared_ptr<ProjDataFromStream> proj_data_ptr =
    new ProjDataFromStream(new_data_info_ptr,sino_stream);

  write_basic_interfile_PDFS_header(output_file_name, *proj_data_ptr);
  cerr << "Output will be written to " << output_file_name 
       << " and its Interfile header\n";

  
  const int dispstart = 
    ask_num("Display start image ? no (0), yes (1)", 
    0,1,0);

  const int save = 
    ask_num("Save  start images ? no (0), yes (1)",
    0,1,0);   
  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr = 0;   
  VoxelsOnCartesianGrid<float> * vox_image_ptr = 0;


  switch (ask_num("Start image is cuboid (1) or cylinder (2) or on file (3)",1,3,2))
  {
  case 1:
    {
      const float zoom = ask_num("Zoom factor (>1 means smaller voxels)",0.F,10.F,1.F);
      int xy_size = static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom);
      xy_size = ask_num("Number of x,y pixels",3,xy_size*2,xy_size);
      image_sptr = vox_image_ptr =
        new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()),
                                         zoom,
                                         CartesianCoordinate3D<float>(0,0,0),
                                         xy_size);      
      fill_cuboid(*vox_image_ptr);
      break;
    }
  case 2:
    {
      const float zoom = ask_num("Zoom factor (>1 means smaller voxels)",0.F,10.F,1.F);
      int xy_size = static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom);
      xy_size = ask_num("Number of x,y pixels",3,xy_size*2,xy_size);
      image_sptr = vox_image_ptr =
        new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()),
                                         zoom,
                                         CartesianCoordinate3D<float>(0,0,0),
                                         xy_size);
      fill_cylinder(*vox_image_ptr);
      break;
    }
  case 3:
    {
      char filename[max_filename_length];
      
      ask_filename_with_extension(filename, "Input file name ?", ".hv");
      
      image_sptr =
        DiscretisedDensity<3,float>::read_from_file(filename);
      vox_image_ptr = dynamic_cast<VoxelsOnCartesianGrid<float> *> (image_sptr.get());
      
      break;
    }
  }

  const float z_origin = 
    ask_num("Shift z-origin (in pixels)", 
             -vox_image_ptr->get_length()/2,
             vox_image_ptr->get_length()/2,
             0) *vox_image_ptr->get_voxel_size().z();
  
  vox_image_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));
  // use shared_ptr such that it cleans up automatically
  shared_ptr<ForwardProjectorByBin> forw_projector_ptr =
    ForwardProjectorByBin::ask_type_and_parameters();
  forw_projector_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
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
    write_basic_interfile("test_image", *image_sptr);
  }
    
  list<ViewSegmentNumbers> already_processed;

  if (ask("Do full forward projection ?", true))
  {    

    CPUTimer timer;
    timer.reset();
    timer.start();

    
    do_segments(*vox_image_ptr, *proj_data_ptr,
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
    for (int segment_num = proj_data_ptr->get_min_segment_num(); 
         segment_num <= proj_data_ptr->get_max_segment_num(); 
         ++segment_num)
    {
      const SegmentByView<float> segment = 
        proj_data_ptr->get_empty_segment_by_view(segment_num, false);
      if (!(proj_data_ptr->set_segment(segment) == Succeeded::yes))
        warning("Error set_segment %d\n", segment_num);            
    }
    do
    {
      CPUTimer timer;
      timer.reset();
      timer.start();
      
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
	    const int start_segment_num, const int end_segment_num,
	    const int start_view, const int end_view,
	    const int start_tangential_pos_num, const int end_tangential_pos_num,
	    ForwardProjectorByBin& forw_projector,
            const bool disp)
{
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    forw_projector.get_symmetries_used()->clone();  
  
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
      
      RelatedViewgrams<float> viewgrams = 
        proj_data.get_empty_related_viewgrams(vs, symmetries_sptr,false);
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
  
  const double zc = 
    ask_num("Centre Z coordinate",     
	    (double)image.get_min_z(), (double)image.get_max_z(), 
	    (image.get_min_z()+ image.get_max_z())/2.);

  
  const double Rcyl = 
    ask_num("Radius (pixels)", 
	    .5, (image.get_max_x()- image.get_min_x())/2.,
	    (image.get_max_x()- image.get_min_x())/4.);
  
  // Max length is num_planes+1 because of edges of voxels
  const double Lcyl = 
    ask_num("Length (planes)", 1., (image.get_max_z()- image.get_min_z())+1.,
	    (image.get_max_z()- image.get_min_z())+1.);
  
  
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
	plane[y][x] = voxel_value*value/(num_samples*num_samples); 
      }
    
  for (int z=image.get_min_z(); z<=image.get_max_z(); z++)
    {
      // use 2. to make both args of min() and max() double
    float zfactor = (std::min(z+.5, zc+Lcyl/2.) - std::max(z-.5, zc-Lcyl/2.));
      if (zfactor<0) zfactor = 0;
      image[z] = plane;
      image[z] *= zfactor;
    }
    
}

