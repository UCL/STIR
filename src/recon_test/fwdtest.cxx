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

#include "Tomography_common.h"

#include "recon_buildblock/fwdproj.h"
#include "display.h"
// for forward projection timer
//#include "recon_buildblock/timers.h"
#include "interfile.h"
// for ask_filename...
#include "utilities.h"

// for auto_ptr
#include <memory>
#include <iostream>
#include <fstream>

//#ifndef TOMO_NO_NAMESPACES
//using std::fstream;
//using std::endl;
//#endif

USING_NAMESPACE_STD
//USING_NAMESPACE_TOMO


START_NAMESPACE_TOMO

// KT 06/04/2000 use auto_ptr, renamed s3d to proj_data

/******************* Declarations local functions *******************/
auto_ptr<PETSinogramOfVolume> ask_parameters();

static void 
do_segments(const PETImageOfVolume& image, const PETSinogramOfVolume& s3d,
	    const int abs_segment_num, 
	    const int start_view, const int end_view,
	    ostream * out,
	    const int fwdproj_method,
	    const int disp, const int save);
static void 
fill_cuboid(PETImageOfVolume& image);
static void 
fill_cylinder(PETImageOfVolume& image);


/*************************** main *************** *******************/
int 
main(int argc, char *argv[])
{
  auto_ptr<PETSinogramOfVolume> proj_data_ptr;

  if(argc!=2) 
  {
    cerr<<"Usage: " << argv[0] << " [PSOV-file]\n"
        <<"The PSOV-file will be used to get the scanner, mashing etc. details" 
	<< endl; 
  }
  if (argc>2)
    exit(EXIT_FAILURE);
  
  if(argc==2)
  {
    proj_data_ptr = 
      auto_ptr<PETSinogramOfVolume>(new PETSinogramOfVolume(read_interfile_PSOV(argv[1])));
  }
  else
  {
    proj_data_ptr = ask_parameters();
  }

  // make num_bins odd (TODO remove)
  {
    int num_bins = proj_data_ptr->scan_info.get_num_bins();
    if (num_bins%2 == 0)
      num_bins++;
    proj_data_ptr->scan_info.set_num_bins(num_bins);
  }
 
  const int fwdproj_method = ask_num("Which method (0: Approximate, 1: Accurate)",0,1,0);

  int disp = 
    ask_num("Display images ? no (0), end result only (1), start image as well (2)", 
    0,2,1);

  int save = 
    ask_num("Save  images ? no (0), end result only (1), start image as well (2)", 
    0,2,1);

   
  PETImageOfVolume image(proj_data_ptr->scan_info);

  switch (ask_num("Start image is cuboid (1) or cylinder (2) or on file (3)",1,3,2))
  {
  case 1:
    fill_cuboid(image);
    break;
  case 2:
    fill_cylinder(image);
    break;
  case 3:
    ifstream input;
    ask_filename_and_open(
      input,
      "Name of image file to forward project", ".hv", 
      ios::in);  

    image = read_interfile_image(input);
    // TODO remove whenever we don't need odd sizes anymore
    if (image.get_x_size() %2 == 0)
    {
      image.grow(image.get_min_z(),image.get_max_z(),
	         image.get_min_y(), -image.get_min_y(),
		 image.get_min_x(), -image.get_min_x());
    }
    break;
  }

  if (disp==2)
    {
      cerr << "Displaying start image";
      display(image);
    }

  fstream *out = 0;

  if (save==2)
  {
    cerr << "Saving start image to 'test_image'" << endl;
    write_basic_interfile("test_image", image);
  }
    
  if (ask("Do full forward projection ?", true))
  {
    const int max_segment_num_to_process = 
      ask_num("max_segment_num_to_process",0,  proj_data_ptr->get_max_segment(),  proj_data_ptr->get_max_segment());
    if (save)
    {
      vector<int> segment_sequence(2*max_segment_num_to_process+1);
      vector<int> min_ring_diff(2*max_segment_num_to_process+1);
      vector<int> max_ring_diff(2*max_segment_num_to_process+1);
      vector<int> min_r(2*max_segment_num_to_process+1);
      vector<int> max_r(2*max_segment_num_to_process+1);
      segment_sequence[0] = 0;
      for (int s=1; s<= max_segment_num_to_process; s++)
      {
	segment_sequence[2*s-1] = s;
	segment_sequence[2*s] = -s;
      }
      { 
	// VC 5.0 bug: can't define 'int index' in the for loop
	int index = 0;
	for (int * iter = segment_sequence.begin();
 	     iter != segment_sequence.end(); 
	     iter++, index++)
	{
	  min_ring_diff[index] = proj_data_ptr->get_min_ring_difference(*iter);
	  max_ring_diff[index] = proj_data_ptr->get_max_ring_difference(*iter);
	  min_r[index] = 0;
	  max_r[index] = proj_data_ptr->get_num_rings(*iter)-1;
	}
      }
      
      cerr << "Saving in 'fwd_image.hs'" << endl;
      out = new (fstream);
      open_write_binary(*out, "fwd_image.s");
      PETSinogramOfVolume news3d(proj_data_ptr->scan_info, 
	segment_sequence,
	min_ring_diff, 	max_ring_diff,
	min_r, 	max_r,
	0, proj_data_ptr->get_num_views()-1,
	-(proj_data_ptr->get_num_bins()/2), -(proj_data_ptr->get_num_bins()/2)+proj_data_ptr->get_num_bins(),
		      *out, 0L, 
		      PETSinogramOfVolume::SegmentViewRingBin,
		      NumericType::FLOAT,
		      ByteOrder::native);

      write_basic_interfile_PSOV_header("fwd_image.hs",
				       "fwd_image.s",
				       news3d);
    }

   // CPUTimer timer;
   // timer.reset();
   // timer_fp.reset();
   // timer.start();
   // timer_fp.start();

    for (int abs_segment_num=0; 
         abs_segment_num<= max_segment_num_to_process; 
         abs_segment_num++)
    {
      cerr << "  - Processing segment num " << abs_segment_num << endl;
          
      // CL&KT 14/02/2000 added fwdproj_method
      do_segments(image, *proj_data_ptr,
                  abs_segment_num, 0, proj_data_ptr->scan_info.get_num_views()-1,
                  out,  fwdproj_method, disp, save);
    }
    //timer.stop();
   // cerr << timer.value() << " s CPU time"<<endl;
   // cerr << timer_fp.value() << " s CPU time forward projection"<<endl;
    
    if (save)
    {
      out->close();
      delete out;
    }


  }
  else
  {   
    do
    {
     // CPUTimer timer;
     // timer.reset();
     // timer_fp.reset();
     // timer.start();
      //timer_fp.start();
      
      int abs_segment_num = ask_num("Segment number to forward project",
				    0, proj_data_ptr->get_max_segment(), 0);
      
      const int nviews = proj_data_ptr->scan_info.get_num_views();
      cerr << "Special views are at 0, "
	<< nviews/4 <<", " << nviews/2 <<", " << nviews/4*3 << endl;
      int start_view = ask_num("Start view", 0, nviews-1, 0);
      int end_view = ask_num("End   view", 0, nviews-1, start_view);
      
      do_segments(image, *proj_data_ptr, 
	          abs_segment_num, 
	          start_view, end_view,
	          0, 
		  fwdproj_method,
	          disp, save);

     // timer.stop();
     // cerr << timer.value() << " s CPU time"<<endl;
    //  cerr << timer_fp.value() << " s CPU time forward projection"<<endl;   

    }
    while (ask("One more ? ", true));
  }

  return EXIT_SUCCESS;
  
}

/******************* Implementation local functions *******************/
// CL&KT 14/02/2000 added fwdproj_method
void
do_segments(const PETImageOfVolume& image, 
            const PETSinogramOfVolume& proj_data,
	    const int abs_segment_num, 
	    const int start_view, const int end_view,
	    ostream * out,
	    const int fwdproj_method,
	    const int disp, const int save)
{
  const int nviews = proj_data.get_num_views();

  PETSegmentByView 
    segment_pos = proj_data.get_empty_segment_view_copy(abs_segment_num);
  PETSegmentByView 
    segment_neg = proj_data.get_empty_segment_view_copy(-abs_segment_num);
  
  {       
    
    Tensor1D<int> processed_views(0, nviews/4);
    
    for (int view= start_view; view<=end_view; view++)
      { 
	int view_to_process = view % (nviews/2);
	if (view_to_process > nviews/4)
	  view_to_process = nviews/2 - view_to_process;
	if (processed_views[view_to_process])
	  continue;
	processed_views[view_to_process] = 1;
      
	cerr << "Processing view " << view_to_process 
	     << " of segment-pair " <<segment_pos.get_segment_num()
	     << endl;
      
	if (segment_pos.get_segment_num() == 0)
	  {
	    forward_project_2D(image, segment_pos, view_to_process);
	  }
	else
	  {
	    // CL&KT 14/02/2000 added fwdproj_method
	    forward_project(fwdproj_method, image, segment_pos, segment_neg, view_to_process);
	  }
      
      }
    
    cerr << "min and max in segment " << abs_segment_num 
	 << ": " << segment_pos.find_min() 
	 << " " << segment_pos.find_max() << endl;
    if (segment_pos.get_segment_num() != 0)
      cerr << "min and max in segment " << -abs_segment_num 
	   << ": " << segment_neg.find_min() 
	   << " " << segment_neg.find_max() << endl;
    
    if (disp)
      {
	cerr << "Displaying segment " << abs_segment_num << endl;
	display(segment_pos, segment_pos.find_max());
	if (segment_pos.get_segment_num() != 0)
	  {
	    cerr << "Displaying segment " << -abs_segment_num<< endl;
	    display(segment_neg, segment_neg.find_max());
	  }
      }
    if (save && out==0)
      {
	{
	  char* file = "fwdtestpos";
	  cerr <<"  - Saving " << file << endl;
	  write_basic_interfile(file, segment_pos);
	}
	if (segment_pos.get_segment_num() != 0)	    
	  {
	    char* file = "fwdtestneg";
	    cerr <<"  - Saving " << file << endl;
	    write_basic_interfile(file, segment_neg);
	  }
      }

    // TODO replace by set_segment
    if (out != 0)
      {
	segment_pos.write_data(*out);
	if (segment_pos.get_segment_num() != 0)
	  {
	    segment_neg.write_data(*out);
	  }
      }
    
    
    
  }
  
}



void fill_cuboid(PETImageOfVolume& image)
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

void fill_cylinder(PETImageOfVolume& image)
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
  
  Tensor2D<float> plane = image[0];
  
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
      float zfactor = (min(z+.5, zc+Lcyl/2.) - max(z-.5, zc-Lcyl/2.));
      if (zfactor<0) zfactor = 0;
      image[z] = plane;
      image[z] *= zfactor;
    }
    
}

auto_ptr<PETSinogramOfVolume>
ask_parameters()
{
    
  PETScanInfo scan_info;
 
  int scanner_num = 
    ask_num("Enter scanner number (0: RPT, 1:1: 953, 2: 966, 3: GE, 4: ART) ? ", 
            0,4,0);
  switch( scanner_num )
    {
    case 0:
      scan_info = (PETScannerInfo::RPT);
      break;
    case 1:
      scan_info = (PETScannerInfo::E953);
      break;
    case 2:
      scan_info = (PETScannerInfo::E966);
      break;
    case 3:
      scan_info = (PETScannerInfo::Advance);
      break;
    case 4:
      scan_info = (PETScannerInfo::ART);
      break;
    }

  {
    const int new_num_bins = 
      scan_info.get_num_bins() / ask_num("Reduce num_bins by factor", 1,16,1);

    // keep same radius of FOV
    scan_info.set_bin_size(
      (scan_info.get_bin_size()*scan_info.get_num_bins()) / new_num_bins
      );

    scan_info.set_num_bins(new_num_bins); 

    scan_info.set_num_views(
      scan_info.get_num_views()/ ask_num("Reduce num_views by factor", 1,16,1)
      );  

  }    

  
  int span = 1;
  {
    if ( scan_info.get_scanner().type != PETScannerInfo::Advance )
      {
	do 
	  {
	    span = ask_num("Span ", 1, scan_info.get_num_rings()-1, 1);
	  }
	while (span%2==0);
      }
  }



  fstream *out = 0;
  
  return new PETSinogramOfVolume(
    scan_info, 
    span, 
    scan_info.get_scanner().type != PETScannerInfo::Advance ?
    scan_info.get_num_rings()-1 : 11,
    *out, 0UL,
    PETSinogramOfVolume::SegmentViewRingBin,
    NumericType::FLOAT);
}

END_NAMESPACE_TOMO
