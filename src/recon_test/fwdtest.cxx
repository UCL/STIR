//
// $Id$: $Date$
//

#include "pet_common.h" 

#include "recon_buildblock/fwdproj.h"
#include "display.h"
// KT 06/10/98 use forward projection timer
#include "recon_buildblock/timers.h"
// KT 09/10/98 use interfile output
#include "interfile.h"
// KT 21/10/98 include for ask_filename...
#include "utilities.h"


// KT 14/10/98 completely restructed to allow full forward projection

/******************* Declarations local functions *******************/
// KT 21/10/98 made all of these static
static void 
do_segments(const PETImageOfVolume& image, const PETScannerInfo& scanner,
	    const int abs_segment_num, 
	    const int start_view, const int end_view,
	    ostream * out,
	    const int disp, const int save);
// KT 21/10/98 without scanner argument now
static void 
fill_cuboid(PETImageOfVolume& image);
// KT 21/10/98 without scanner argument now
static void 
fill_cylinder(PETImageOfVolume& image);


/*************************** main *************** *******************/
int
main()
{
 
  PETScannerInfo scanner;
 
  int scanner_num = 
    ask_num("Enter scanner number (0: RPT, 1:1: 953, 2: 966, 3: GE) ? ", 0,3,0);
  switch( scanner_num )
    {
    case 0:
      scanner = (PETScannerInfo::RPT);
      break;
    case 1:
      scanner = (PETScannerInfo::E953);
      break;
    case 2:
      scanner = (PETScannerInfo::E966);
      break;
    case 3:
      scanner = (PETScannerInfo::Advance);
      break;
    }


  // KT 02/07/98 new
  {
    scanner.num_bins /= ask_num("Reduce num_bins by factor", 1,16,1); 
    scanner.num_views /= ask_num("Reduce num_views by factor", 1,16,1);  
    // scanner.num_rings /= 1;

    scanner.bin_size = 2* scanner.FOV_radius / scanner.num_bins;
    // scanner.ring_spacing = scanner.FOV_axial / scanner.num_rings;
  }    

  // KT 06/10/98 more options
  int disp = 
    ask_num("Display images ? no (0), end result only (1), start image as well (2)", 
    0,2,1);

  // KT 14/10/98 more options
  int save = 
    ask_num("Save  images ? no (0), end result only (1), start image as well (2)", 
    0,2,1);

 
  Point3D origin(0,0,0);
  Point3D voxel_size(scanner.bin_size,
		     scanner.bin_size,
		     scanner.ring_spacing/2); 
 
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;

  PETImageOfVolume image(Tensor3D<float>( 0,2*scanner.num_rings-2, 
					  -(scanner.num_bins/2), max_bin,
					  -(scanner.num_bins/2), max_bin),
			 origin, voxel_size);

  // KT 21/10/98 allow 2 types of images and input from file
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
    break;
  }

  // KT 06/10/98 only when selected by user
  if (disp==2)
    {
      cerr << "Displaying start image";
      display(Tensor3D<float>(image));
    }
  // KT 14/10/98 new
  if (save==2)
    {
      cerr << "Saving start image to 'test_image'" << endl;
      write_basic_interfile("test_image", image);
    }
  
  // KT 14/10 rewrite using segment numbers
  int max_segment_num = scanner.num_rings-1; 
  // KT 06/10/98 use name
  if (scanner.type == PETScannerInfo::Advance)
    max_segment_num = 11;
  
  if (ask("Do full forward projection ?", true))
  {
    ofstream *out = 0;
    if (save)
    {
      cerr << "Saving in 'fwd_image.scn'" << endl;
      out = new (ofstream);
      open_write_binary(*out, "fwd_image.scn");
    }

    CPUTimer timer;
    timer.restart();
    timer_fp.restart();

    for (int abs_segment_num=0; 
         abs_segment_num<= max_segment_num; 
         abs_segment_num++)
      do_segments(image, scanner,
                  abs_segment_num, 0, scanner.num_views-1,
                  out, disp, save);
    timer.stop();
    cerr << timer.value() << " s CPU time"<<endl;
    cerr << timer_fp.value() << " s CPU time forward projection"<<endl;
    
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
      CPUTimer timer;
      timer.restart();
      timer_fp.restart();
      
      int abs_segment_num = ask_num("Segment number to forward project",
				    0, max_segment_num, 0);
      
      const int nviews = scanner.num_views;
      cerr << "Special views are at 0, "
	<< nviews/4 <<", " << nviews/2 <<", " << nviews/4*3 << endl;
      // KT 02/07/98 allow more views
      int start_view = ask_num("Start view", 0, nviews-1, 0);
      int end_view = ask_num("End   view", 0, nviews-1, start_view);
      
      do_segments(image, scanner, 
	          abs_segment_num, 
	          start_view, end_view,
	          0, 
	          disp, save);

      timer.stop();
      cerr << timer.value() << " s CPU time"<<endl;
      cerr << timer_fp.value() << " s CPU time forward projection"<<endl;   

    }
    // KT 06/10/98 use ask()
    while (ask("One more ? ", true));
  }
  return 0;
  
}

/******************* Implementation local functions *******************/
void
do_segments(const PETImageOfVolume& image, const PETScannerInfo& scanner,
	    const int abs_segment_num, 
	    const int start_view, const int end_view,
	    ostream * out,
	    const int disp, const int save)
{
  
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;
  const int nviews = scanner.num_views;
  
  // KT 07/10/98 now handle segment 0 correctly
  
  int pos_max_ring_diff = abs_segment_num;
  int pos_min_ring_diff = abs_segment_num;
  int num_rings_this_segment =
    scanner.num_rings - abs_segment_num;
  
  // GEAdvance has different segment number etc
  // KT 06/10/98 use name
  if (scanner.type == PETScannerInfo::Advance)
    {
      if (abs_segment_num == 0)
	{
	  pos_max_ring_diff = 1;
	  pos_min_ring_diff = -1;
	  num_rings_this_segment = 2*scanner.num_rings-1;
	}
      else
	{ 
	  pos_max_ring_diff++;
	  pos_min_ring_diff++;
	  num_rings_this_segment--;
	}
    
    }
  
  
  PETSegmentByView 
    segment_pos(
		Tensor3D<float>(0, scanner.num_views-1, 
				0, num_rings_this_segment - 1,
				-scanner.num_bins/2, max_bin),
		&scanner,
		abs_segment_num,
		pos_min_ring_diff,
		pos_max_ring_diff);
  
  PETSegmentByView 
    segment_neg(
		Tensor3D<float>(0, scanner.num_views-1, 
				0, num_rings_this_segment - 1,
				-scanner.num_bins/2, max_bin),
		&scanner,
		-abs_segment_num, 
		-pos_max_ring_diff, 
		-pos_min_ring_diff); 
  
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
      
	// KT 03/09/98 add output of segment number
	cerr << "Processing view " << view_to_process 
	     << " of segment-pair " <<segment_pos.get_segment_num()
	     << endl;
      
	if (segment_pos.get_segment_num() == 0)
	  {
	    forward_project_2D(image, segment_pos, view_to_process);
	  }
	else
	  {
	    forward_project(image, segment_pos, segment_neg, view_to_process);
	  }
      
      }
    
    // KT 14/10/98 added number
    cerr << "min and max in segment " << abs_segment_num 
	 << ": " << segment_pos.find_min() 
	 << " " << segment_pos.find_max() << endl;
    // KT 14/10/98 output segment_neg as well
    if (segment_pos.get_segment_num() != 0)
      cerr << "min and max in segment " << -abs_segment_num 
	   << ": " << segment_neg.find_min() 
	   << " " << segment_neg.find_max() << endl;
    
    if (disp)
      {
	// KT 14/10/98 added number
	cerr << "Displaying segment " << abs_segment_num << endl;
	display(segment_pos, segment_pos.find_max());
	// KT 14/10/98 only when non-zero segment
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
	  // KT 09/10/98 use interfile output
	  write_basic_interfile(file, segment_pos);
	}
	// KT 14/10/98 only when non-zero segment
	if (segment_pos.get_segment_num() != 0)	    
	  {
	    char* file = "fwdtestneg";
	    cerr <<"  - Saving " << file << endl;
	    // KT 09/10/98 use interfile output
	    write_basic_interfile(file, segment_neg);
	  }
      }
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



// KT 21/10/98 use dimensions from the image itself
void fill_cuboid(PETImageOfVolume& image)
{
  
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
	image[z][y][x] = 1; 
}

// KT 21/10/98 use dimensions from the image itself
void fill_cylinder(PETImageOfVolume& image)
{
 
  // KT 21/10/98 made double
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
	plane[y][x] = value/(num_samples*num_samples); 
      }
    
  for (int z=image.get_min_z(); z<=image.get_max_z(); z++)
    {
      // KT 09/10/98 changed 2 -> 2. to make both args of min() and max() double
      float zfactor = (min(z+.5, zc+Lcyl/2.) - max(z-.5, zc-Lcyl/2.));
      if (zfactor<0) zfactor = 0;
      image[z] = plane;
      image[z] *= zfactor;
    }
    
}
