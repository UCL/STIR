//
//

/*!

  \file
  \ingroup recontest
  \brief Testing programme for forward projection

  \author Kris Thielemans
  \author PARAPET project


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
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2003, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
//#include "stir/display.h"
#include "stir/IO/interfile.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfo.h"
// for ask_filename...
#include "stir/utilities.h"
#include "stir/IndexRange3D.h"
#include "stir/RelatedViewgrams.h"
#include "stir/SegmentByView.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CPUTimer.h"
#include <fstream>



USING_NAMESPACE_STIR
USING_NAMESPACE_STD


/******************* Declarations local functions *******************/

static void 
do_segments(const VoxelsOnCartesianGrid<float>& image, ProjData& s3d,
            const int min_z, const int max_z,
            const int min_y, const int max_y,
            const int min_x, const int max_x,
            ProjMatrixByDensel&,
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
      int z_size = 2*proj_data_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()-1;
      z_size = ask_num("Number of z pixels",1,1000,z_size);
      image_sptr = vox_image_ptr =
        new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()),
                                         zoom,
                                         CartesianCoordinate3D<float>(0,0,0),
                                         Coordinate3D<int>(z_size,xy_size,xy_size));
      
      fill_cuboid(*vox_image_ptr);
      break;
    }
  case 2:
    {
      const float zoom = ask_num("Zoom factor (>1 means smaller voxels)",0.F,10.F,1.F);
      int xy_size = static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom);
      xy_size = ask_num("Number of x,y pixels",3,xy_size*2,xy_size);
      int z_size = 2*proj_data_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()-1;
      z_size = ask_num("Number of z pixels",1,1000,z_size);
      image_sptr = vox_image_ptr =
        new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()),
                                         zoom,
                                         CartesianCoordinate3D<float>(0,0,0),
                                         Coordinate3D<int>(z_size,xy_size,xy_size));
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
  shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
    new ProjMatrixByDenselUsingRayTracing;
/*  do 
    {
    proj_matrix_ptr=
      ProjMatrixByDensel::ask_type_and_parameters();
    }
  while (proj_matrix_ptr.use_count()==0);
*/
  proj_matrix_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
			     image_sptr);
  cerr << proj_matrix_ptr->parameter_info();

  if (dispstart)
    {
      cerr << "Displaying start image";
      //display(*image_sptr, image_sptr->find_max());
    }
  

  if (save)
  {
    cerr << "Saving start image to 'test_image'" << endl;
    write_basic_interfile("test_image", *image_sptr);
  }
    
  
//  if (ask("Do full forward projection ?", true))
  //do
  {    
    const int min_z = ask_num("Min z", vox_image_ptr->get_min_z(), vox_image_ptr->get_max_z(), vox_image_ptr->get_min_z());
    const int max_z = ask_num("Max z", min_z, vox_image_ptr->get_max_z(), vox_image_ptr->get_max_z());
    const int min_y = ask_num("Min y", vox_image_ptr->get_min_y(), vox_image_ptr->get_max_y(), vox_image_ptr->get_min_y());
    const int max_y = ask_num("Max y", min_y, vox_image_ptr->get_max_y(), vox_image_ptr->get_max_y());
    const int min_x = ask_num("Min x", vox_image_ptr->get_min_x(), vox_image_ptr->get_max_x(), vox_image_ptr->get_min_x());
    const int max_x = ask_num("Max x", min_x, vox_image_ptr->get_max_x(), vox_image_ptr->get_max_x());
    
    CPUTimer timer;
    timer.reset();
    timer.start();

    
    do_segments(*vox_image_ptr, *proj_data_ptr,
                min_z, max_z,
                min_y, max_y,
                min_x, max_x,
                *proj_matrix_ptr,
                false);
    
    timer.stop();
    cerr << timer.value() << " s CPU time"<<endl;
    

  }
  //  while (ask("One more ? ", false));
  

  return EXIT_SUCCESS;
  
}

/******************* Implementation local functions *******************/
void
do_segments(const VoxelsOnCartesianGrid<float>& image, 
            ProjData& proj_data,
            const int min_z, const int max_z,
            const int min_y, const int max_y,
            const int min_x, const int max_x,
	    ProjMatrixByDensel& proj_matrix,
            const bool disp)
{
  const int start_segment_num = proj_data.get_min_segment_num();
  const int end_segment_num = proj_data.get_max_segment_num();

  VectorWithOffset<SegmentByView<float> *> all_segments(start_segment_num, end_segment_num);

  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    all_segments[segment_num] = new SegmentByView<float>(proj_data.get_empty_segment_by_view(segment_num));

  ProjMatrixElemsForOneDensel probs;
  for (int z = min_z; z<= max_z; ++z)
  {
    //std::cerr << "z "<< z<< std::endl;
    for (int y = min_y; y<= max_y; ++y)
    {
      for (int x = min_x; x<= max_x; ++x)
      {
        if (image[z][y][x] == 0)
          continue;
        Densel densel(z,y,x);
        proj_matrix.get_proj_matrix_elems_for_one_densel(probs, densel);
        for (ProjMatrixElemsForOneDensel::const_iterator element_ptr = probs.begin();
             element_ptr != probs.end();
             ++element_ptr)
        {
          if (element_ptr->axial_pos_num()<= proj_data.get_max_axial_pos_num(element_ptr->segment_num()) &&
              element_ptr->axial_pos_num()>= proj_data.get_min_axial_pos_num(element_ptr->segment_num()))
            (*all_segments[element_ptr->segment_num()])[element_ptr->view_num()][element_ptr->axial_pos_num()][element_ptr->tangential_pos_num()] +=
              image[z][y][x] * element_ptr->get_value();
        }
      }
    }
  }

  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  { 
    if (!(proj_data.set_segment(*all_segments[segment_num]) == Succeeded::yes))
        error("Error set_segment\n");            
    delete all_segments[segment_num];
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

