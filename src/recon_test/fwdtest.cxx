//
// $Id$: $Date$
//

#include <fstream.h>

#include "pet_common.h" 

// KT 09/06/98 don't use glgtime.h anymore
//#include "PROMIS/glgtime.h"
#include "recon_buildblock/fwdproj.h"
#include "display.h"
#include "Timer.h"

// KT 09/06/98 don't use glgtime.h anymore, so declarations of Vor_proj etc removed

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

  int disp = ask_num("Display images ? (0: No, 1: Yes)", 0, 1, 1);

  int save = ask_num("Save  images (0: No, 1: Yes) ? ", 0,1,0);

 
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

 


  while(true)
    {
      int max_ring_diff = scanner.num_rings-1; // DB another 'Advanced' bug fixed
      if (scanner_num == 3) // GE Advance
	max_ring_diff = 11;

      int abs_ave_ring_diff = ask_num("Ring difference to forward project (0 to end)",
				      0, max_ring_diff, 0);
      if (abs_ave_ring_diff == 0)
	break;

      //      int abs_segment_num = segment_num >= 0 ? segment_num : -segment_num;
      int abs_segment_num = abs_ave_ring_diff;

      //DB GEAdvance has different numbers: segment number and average_ring diff

      if (scanner_num == 3) // GE Advance
	if (abs_ave_ring_diff == 1)
	  {
	    PETerror("ring difference 1 not allowed for GE Advance\n");
	    break;
	  }
	else
	  abs_segment_num--;

      PETSegmentByView 
	segment_pos(
		    Tensor3D<float>(0, scanner.num_views-1, 
				    0, scanner.num_rings-1 - abs_ave_ring_diff, // DB see above
				    -scanner.num_bins/2, max_bin),
		    &scanner,
		    abs_segment_num, // DB ABS
		    abs_ave_ring_diff, // min ring diff
		    abs_ave_ring_diff); // max ring diff

      PETSegmentByView 
	segment_neg(
		    Tensor3D<float>(0, scanner.num_views-1, 
				    0, scanner.num_rings-1 - abs_ave_ring_diff, // DB see above
				    -scanner.num_bins/2, max_bin),
		    &scanner,
		    -abs_segment_num, // DB ABS!!!!!!!!!!!!!!!
		    -abs_ave_ring_diff, // min ring diff
		    -abs_ave_ring_diff); // max ring diff

      { // forward
	segment_pos.fill(0);
	segment_neg.fill(0);
	  

	const int nviews = scanner.num_views;
	cerr << "Special views are at 0, "
	     << nviews/4 <<", " << nviews/2 <<", " << nviews/4*3 << endl;
	int view = ask_num("View", 0, nviews-1, 0);
        int view_to_process = view % (nviews/2);
	if (view_to_process > nviews/4)
	  view_to_process = nviews/2 - view_to_process;
	cerr << endl
	     << "Symmetry related view which will be passed to forward_project :"
	     << view_to_process << endl;
	cerr << endl
	     << " Image filling" << endl;
	cerr << endl << endl;

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
	  
	if (disp)
	  {
	    cerr << "Displaying start image";
	    display(Tensor3D<float>(image));
	  }
	  
        CPUTimer timer;
	timer.restart();
	forward_project(image, segment_pos, segment_neg, view_to_process);


	timer.stop();
	cerr << timer.value() << " s CPU time"<<endl;

	cerr << "min and max in segment_pos " << segment_pos.find_min() 
	     << " " << segment_pos.find_max() << endl;

	for (int view=0; view< segment_pos.get_num_views(); view++)
	  {
	    cerr << "min and max for view " << view 
		 << ": " << segment_pos[view].find_min() 
		 << ", " << segment_pos[view].find_max() << endl;
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
	    char* file = "fwdtestpos.img";
	    cerr <<"  - Saving " << file << endl;
	    ofstream output;
	    open_write_binary(output, file);

	    segment_pos.write_data(output);
	  }
	  {
	    char* file = "fwdtestneg.img";
	    cerr <<"  - Saving " << file << endl;
	    ofstream output;
	    open_write_binary(output, file);

	    segment_neg.write_data(output);
	  }
	}

      }

    }

  return 0;

}


