//
// $Id$: $Date$
//

/*!

  \file

  \brief Testing programme for forward projection

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$

  This (ugly) programme allows forward projection of a few segments/views
  only. 
  It can also be used to forward project into the full data set. 

  \warning This is not intended as a user-friendly forward projector, although
  it would be easy to make one starting from this code.

  Usage:
  \verbatim
  fwdtest [proj_data_file]
  \endverbatim
  The proj_data_file will be used to get the scanner, mashing etc. details
  (its data will \e not be used, nor will it be overwritten).
  If no proj_data_file is given, some questions are asked to use 'standard'
  characteristics.        
*/
#include "recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "display.h"
#include "interfile.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfo.h"
// for ask_filename...
#include "utilities.h"
#include "IndexRange3D.h"
#include "RelatedViewgrams.h"
#include <fstream>


// for auto_ptr
#include <memory>

USING_NAMESPACE_TOMO
USING_NAMESPACE_STD


/******************* Declarations local functions *******************/
auto_ptr<ProjDataFromStream> ask_parameters();

static void 
do_segments(const VoxelsOnCartesianGrid<float>& image, ProjDataFromStream& s3d,
	    const int abs_segment_num, 
	    const int start_view, const int end_view,
	    ForwardProjectorByBin*,
	    const int disp, const int save);
static void 
fill_cuboid(VoxelsOnCartesianGrid<float>& image);
static void 
fill_cylinder(VoxelsOnCartesianGrid<float>& image);



/*************************** main ***********************************/
int 
main(int argc, char *argv[])
{

/*  if(argc!=2) 
  {
    cerr<<"Usage: " << argv[0] << " [PSOV-file]\n"
        <<"The PSOV-file will be used to get the scanner, mashing etc. details" 
	<< endl; 
  }
  if (argc>2)
    exit(EXIT_FAILURE);
*/

 // if(argc==2)
 // {
   shared_ptr<ProjData> proj_data_ptr = 
     ProjData::read_from_file(argv[1]);
  //}
 // else
// {
 //   proj_data_ptr = ask_parameters();
  //}

 // auto_ptr<ProjDataFromStream> proj_data_ptr(ProjDataFromStream::ask_parameters());
 // }

#if 1
  const ProjDataInfo* data_info = proj_data_ptr->get_proj_data_info_ptr();
  string header_name = argv[1];
  header_name += "_copy.hs"; 
  string raw_data = argv[1];
   raw_data+=" _copy.prj";
  ProjDataInfo* new_data_info= data_info->clone();
 
  
  sino_stream = new fstream (raw_data.c_str(), ios::out|ios::binary);
  if (!sino_stream->good())
  {
    error("fwdtest: error opening file %s\n",raw_data);
  }

  
   ProjDataFromStream new_data(new_data_info,sino_stream);
 
#endif

#if 1

  // make num_bins odd (TODO remove)
  {
    int num_bins = new_data.get_proj_data_info_ptr()->get_num_tangential_poss();
    if (num_bins%2 == 0)
      num_bins++;
    new_data_info->set_num_tangential_poss(num_bins);
  }
#endif

  const int fwdproj_method = ask_num("Which method (0: Approximate, 1: Accurate)",0,1,0);

  int disp = 
    ask_num("Display images ? no (0), end result only (1), start image as well (2)", 
    0,2,1);

  int save = 
    ask_num("Save  images ? no (0), end result only (1), start image as well (2)", 
    0,2,1);

   

  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr; 
  
  VoxelsOnCartesianGrid<float> * vox_image_ptr;
     

  switch (ask_num("Start image is cuboid (1) or cylinder (2) or on file (3)",1,3,2))
  {
  case 1:
    image_sptr = vox_image_ptr =
     new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()));

    fill_cuboid(*vox_image_ptr);
    break;
  case 2:
    image_sptr = vox_image_ptr =
     new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()));
    fill_cylinder(*vox_image_ptr);
    break;
  case 3:
  
   char filename[max_filename_length];
   
    ask_filename_with_extension(filename, "Input file name ?", ".hv");

    image_sptr =
      DiscretisedDensity<3,float>::read_from_file(filename);
    vox_image_ptr = dynamic_cast<VoxelsOnCartesianGrid<float> *> (image_sptr.get());

    // TODO remove whenever we don't need odd sizes anymore
    if (vox_image_ptr->get_x_size() %2 == 0)
    {
      vox_image_ptr->grow(IndexRange3D(vox_image_ptr->get_min_z(),vox_image_ptr->get_max_z(),
	         vox_image_ptr->get_min_y(), -vox_image_ptr->get_min_y(),
		 vox_image_ptr->get_min_x(), -vox_image_ptr->get_min_x()));
    }
    break;
  }

  ForwardProjectorByBinUsingRayTracing * forw_projector_ptr = 
    new ForwardProjectorByBinUsingRayTracing(proj_data_ptr->get_proj_data_info_ptr()->clone(), 
        image_sptr);
    
  
  if (disp==2)
    {
      cerr << "Displaying start image";
      display(*vox_image_ptr);
    }
  

  if (save==2)
  {
    cerr << "Saving start image to 'test_image'" << endl;
    write_basic_interfile("test_image", *vox_image_ptr);
  }
    
  if (ask("Do full forward projection ?", true))
  {
    const int max_segment_num_to_process = 
      ask_num("max_segment_num_to_process",0,  proj_data_ptr->get_max_segment_num(),  proj_data_ptr->get_max_segment_num());
    

    CPUTimer timer;
    timer.reset();
    timer.start();

    for (int abs_segment_num=0; 
         abs_segment_num<= max_segment_num_to_process; 
         abs_segment_num++)
    {
      cerr << "  - Processing segment num " << abs_segment_num << endl;
          
      do_segments(*vox_image_ptr, new_data/*,proj_data_ptr*/,
                  abs_segment_num, 0, proj_data_ptr->get_proj_data_info_ptr()->get_num_views()-1,
                  forw_projector_ptr, disp, save);
			   


    }
    timer.stop();
    cerr << timer.value() << " s CPU time"<<endl;


  }
  else
  {   
    do
    {
      CPUTimer timer;
      timer.reset();
      timer.start();
      
      int abs_segment_num = ask_num("Segment<float> number to forward project",
				    0, proj_data_ptr->get_max_segment_num(), 0);
      
      const int nviews = proj_data_ptr->get_proj_data_info_ptr()->get_num_views();
      cerr << "Special views are at 0, "
	<< nviews/4 <<", " << nviews/2 <<", " << nviews/4*3 << endl;
      int start_view = ask_num("Start view", 0, nviews-1, 0);
      int end_view = ask_num("End   view", 0, nviews-1, start_view);
      
      do_segments(*vox_image_ptr,new_data,/*proj_data_ptr*/ 
	          abs_segment_num, 
	          start_view, end_view,
	          forw_projector_ptr,
	          disp, save);
      

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
            ProjDataFromStream& proj_data,
	    const int abs_segment_num, 
	    const int start_view, const int end_view,
	    ForwardProjectorByBin* forw_projector_ptr,
	    const int disp, const int save)
{
  const int nviews = proj_data.get_num_views();
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr =
    forw_projector_ptr->get_symmetries_used()->clone();

  {       
    
    Array<1,int> processed_views(0, nviews/4);
    
    for (int view= start_view; view<=end_view; view++)
      { 
	int view_to_process = view % (nviews/2);
	if (view_to_process > nviews/4)
	  view_to_process = nviews/2 - view_to_process;
	if (processed_views[view_to_process])
	  continue;
	processed_views[view_to_process] = 1;
      
	cerr << "Processing view " << view_to_process 
	     << " of segment-pair " <<abs_segment_num
	     << endl;

	const ViewSegmentNumbers view_segmnet_num(view_to_process,abs_segment_num);
	RelatedViewgrams<float> viewgrams = 
	  proj_data.get_proj_data_info_ptr()->
	  get_empty_related_viewgrams(view_segmnet_num,
	    symmetries_sptr);
         forw_projector_ptr->forward_project(viewgrams, image);	  
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
  
  const double zc = 
    ask_num("Centre Z coordinate",     
	    (double)image.get_min_z(), (double)image.get_max_z(), 
	    (image.get_min_z()+ image.get_max_z())/2.);

  
  const double Rcyl = 
    ask_num("Radius", 
	    .5, (image.get_max_x()- image.get_min_x())/2.,
	    (image.get_max_x()- image.get_min_x())/4.);
  
  // Max length is num_planes+1 because of edges of voxels
  const double Lcyl = 
    ask_num("Length", 2., (image.get_max_z()- image.get_min_z())+1.,
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

/*
auto_ptr<ProjDataFromStream>
ask_parameters()
{
    
  ProjDataInfo scan_info;
 
  int scanner_num = 
    ask_num("Enter scanner number (0: RPT, 1:1: 953, 2: 966, 3: GE, 4: ART) ? ", 
            0,4,0);
  switch( scanner_num )
    {
    case 0:
      scan_info = (Scanner::RPT);
      break;
    case 1:
      scan_info = (Scanner::E953);
      break;
    case 2:
      scan_info = (Scanner::E966);
      break;
    case 3:
      scan_info = (Scanner::Advance);
      break;
    case 4:
      scan_info = (Scanner::ART);
      break;
    }

  {
    const int new_num_bins = 
      proj_data_info_ptr->get_num_tangential_poss() / ask_num("Reduce num_bins by factor", 1,16,1);

    // keep same radius of FOV
    proj_data_info_ptr->set_bin_size(
      (proj_data_info_ptr->get_bin_size()*proj_data_info_ptr->get_num_tangential_poss()) / new_num_bins
      );

    proj_data_info_ptr->set_num_bins(new_num_bins); 

    proj_data_info_ptr->set_num_views(
      proj_data_info_ptr->get_num_views()/ ask_num("Reduce num_views by factor", 1,16,1)
      );  

  }    

  
  int span = 1;
  {
    if ( proj_data_info_ptr->get_scanner().type != Scanner::Advance )
      {
	do 
	  {
	    span = ask_num("Span ", 1, proj_data_info_ptr->get_num_axial_poss()-1, 1);
	  }
	while (span%2==0);
      }
  }



  fstream *out = 0;
  
  return new ProjDataFromStream(
    scan_info, 
    span, 
    proj_data_info_ptr->get_scanner().type != Scanner::Advance ?
    proj_data_info_ptr->get_num_axial_poss()-1 : 11,
    *out, 0UL,
    ProjDataFromStream::SegmentViewRingBin,
    NumericType::FLOAT);
}
*/
