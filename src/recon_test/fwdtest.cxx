//
// $Id$: $Date$
//

#include <fstream.h>

#include "pet_common.h" 

#include "recon_buildblock/fwdproj.h"
#include "display.h"
// KT 06/10/98 use forward projection timer
#include "recon_buildblock/timers.h"
// KT 09/10/98 use interfile output
#include "interfile.h"


// KT 07/10/98 new, isolated from main()
void fill_cuboid(PETImageOfVolume& image, const PETScannerInfo& scanner)
{
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;
  
  const int nbins = scanner.num_bins;

  int xs = ask_num("Start X coordinate", -nbins/2, max_bin, 0);
  int ys = ask_num("Start Y coordinate", -nbins/2, max_bin, 0);
  int zs = ask_num("Start Z coordinate", 0, 2*scanner.num_rings-2, 0);
  
  int xe = ask_num("End X coordinate", -nbins/2, max_bin, xs);
  int ye = ask_num("End Y coordinate", -nbins/2, max_bin, ys);
  int ze = ask_num("End Z coordinate", 0, 2*scanner.num_rings-2, zs);
  
  
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
  
// KT 07/10/98 new, isolated from main()
void fill_cylindroid(PETImageOfVolume& image, const PETScannerInfo& scanner)
{
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;
  
  const int nbins = scanner.num_bins;

  const int xc = 
    ask_num("Centre X coordinate", -nbins/2, max_bin, 0);
  const int yc = 
    ask_num("Centre Y coordinate", -nbins/2, max_bin, 0);
  const int zc = 
    ask_num("Centre Z coordinate", 0, 2*scanner.num_rings-2, scanner.num_rings-1);
  
  const float Rcyl = 
    ask_num("Radius", .5F, nbins/2.F, nbins/4.F);
  const float Lcyl = 
    ask_num("Length", 2.F, 2*scanner.num_rings-2.F, 2*scanner.num_rings-2.F);
  
  
  cerr << "Centre coordinate: (x,y,z) = (" 
       << xc << ", " << yc << ", " << zc  
       << ")" << endl;
  cerr << "Radius = " << Rcyl << ", Length = " << Lcyl << endl; 
  
  const int num_samples = 
    ask_num("With how many points do I sample each unit distance ?",1,100,5);
  
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

int
main()
{
 
  PETScannerInfo scanner;
 
  int scanner_num = ask_num("Enter scanner number (0: RPT, 1:1: 953, 2: 966, 3: GE) ? ", 0,3,0);
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
    ask_num("Display images ? no (0), end result only (1), start image as well (2)", 0,2,1);

  // KT 06/10/98 use ask
  bool save = ask("Save  images ? ", false);

 
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

  // KT 06/10/98 allow 2 types of images
  if (ask_num("Start image is cuboid (1) or cylindroid (2)",1,2,2)==1)
    fill_cuboid(image, scanner);
  else
    fill_cylindroid(image, scanner);


  // KT 06/10/98 only when selected by user
  if (disp==2)
    {
      cerr << "Displaying start image";
      display(Tensor3D<float>(image));
    }
	  


  do
    {
      int max_ring_diff = scanner.num_rings-1; 
      // KT 06/10/98 use name
      if (scanner.type == PETScannerInfo::Advance)
	max_ring_diff = 11;

      int abs_segment_num = ask_num("Ring difference to forward project",
				      0, max_ring_diff, 0);

      // KT 07/10/98 now handle segment 0 correctly

      int pos_max_ring_diff = abs_segment_num;
      int pos_min_ring_diff = abs_segment_num;
      int num_rings_this_segment =
	scanner.num_rings - abs_segment_num;

      // GEAdvance has different segment number etc

      // KT 06/10/98 use name
      if (scanner.type == PETScannerInfo::Advance)
      {
	if (abs_segment_num == 1)
	{
	  PETerror("ring difference 1 not allowed for GE Advance\n");
	  break;
	}
	if (abs_segment_num == 0)
	{
	  pos_max_ring_diff = 1;
	  pos_min_ring_diff = -1;
	  num_rings_this_segment = 2*scanner.num_rings-1;
	}
	else
	  abs_segment_num--;
	
      }

      bool use_2D_forwardprojection =  
	 (ask_num("Use 2D forward projection (2) or 3D for segment 0 (3) ? ",
		    2,3,2) == 2);


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

	const int nviews = scanner.num_views;
	cerr << "Special views are at 0, "
	     << nviews/4 <<", " << nviews/2 <<", " << nviews/4*3 << endl;
	// KT 02/07/98 allow more views
	int start_view = ask_num("Start view", 0, nviews-1, 0);
	int end_view = ask_num("End   view", 0, nviews-1, start_view);
        CPUTimer timer;
	timer.restart();
	// KT 06/10/98 added forward projection timer
	timer_fp.restart();

	
	Tensor1D<int> processed_views(0, nviews/4);

	for (int view= start_view; view<=end_view; view++)
	  { 
	    int view_to_process = view % (nviews/2);
	    if (view_to_process > nviews/4)
	      view_to_process = nviews/2 - view_to_process;
	    if (processed_views[view_to_process])
	      continue;
	    processed_views[view_to_process] = 1;

	    cerr << "Symmetry related view which will be passed to forward_project :"
		 << view_to_process << endl;

	    if (segment_pos.get_segment_num() == 0 && use_2D_forwardprojection)
	      {
		forward_project_2D(image, segment_pos, view_to_process);
	      }
	    else
	      {
		forward_project(image, segment_pos, segment_neg, view_to_process);
	      }

	  }

	timer.stop();
	cerr << timer.value() << " s CPU time"<<endl;
	// KT 08/10/98 added print out of backprojection timer
	cerr << timer_fp.value() << " s CPU time forward projection"<<endl;


	cerr << "min and max in segment_pos " << segment_pos.find_min() 
	     << " " << segment_pos.find_max() << endl;

	// KT 06/10/98 added {} for VC++
	{
	  for (int view=0; view< segment_pos.get_num_views(); view++)
	  {
	    cerr << "min and max for view " << view 
		 << ": " << segment_pos[view].find_min() 
		 << ", " << segment_pos[view].find_max() << endl;
	  }
	}

	if (disp)
	  {
	    cerr << "Displaying positive segment" << endl;
	    display(Tensor3D<float>(segment_pos), segment_pos.find_max());
	    cerr << "Displaying negative segment" << endl;
	    display(Tensor3D<float>(segment_neg), segment_neg.find_max());
	  }
        if (save)
	{
	  {
	    char* file = "fwdtestpos";
	    cerr <<"  - Saving " << file << endl;
	    // KT 09/10/98 use interfile output
	    write_basic_interfile(file, segment_pos);
	  }
	  {
	    char* file = "fwdtestneg";
	    cerr <<"  - Saving " << file << endl;
	    // KT 09/10/98 use interfile output
	    write_basic_interfile(file, segment_neg);
	  }
	}

      }

    }
    // KT 06/10/98 use ask()
    while (ask("One more ? ", true));

  return 0;

}


