//
// $Id$: $Date$
//

#include "pet_common.h" 


#include "PROMIS/glgtime.h"
#include "recon_buildblock/fwdproj.h"
#include "display.h"
#include "Timer.h"

#ifdef _TIMER_
#if !defined(_AIX_) && !defined(_SUN_)
clock_t Vor_Proj,Vor_Filter,Vor_Rueckproj,Vor_Rest;
clock_t Start_time;
time_t Vor_diskio;
#else
long Vor_Proj,Vor_Filter,Vor_Rueckproj,Vor_Rest;
long Start_time;
time_t Vor_diskio;
#endif
#endif
double          Rueckprojzeit = 0.0, Projzeit = 0.0;
double          Filterzeit = 0.0, Diskiozeit = 0.0, Restzeit = 0.0;
double          Total_time = 0.0;




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
      int segment_num = ask_num("Ring difference to forward project to (0 to end)",
				-scanner.num_rings+1, scanner.num_rings-1, 0);
      if (segment_num == 0)
	break;

      int abs_segment_num = segment_num >= 0 ? segment_num : -segment_num;

      PETSegmentByView 
	segment_pos(
		    Tensor3D<float>(0, scanner.num_views-1, 
				    0, scanner.num_rings-1 - abs_segment_num,
				    -scanner.num_bins/2, max_bin),
		    &scanner,
		    abs_segment_num);
      PETSegmentByView 
	segment_neg(
		    Tensor3D<float>(0, scanner.num_views-1, 
				    0, scanner.num_rings-1 - abs_segment_num,
				    -scanner.num_bins/2, max_bin),
		    &scanner,
		    -abs_segment_num);

      { // forward
	segment_pos.fill(0);
	segment_neg.fill(0);
	  
	cerr << "Coordinates of start z,y,x and end z,y,x" << endl;
	int zs,ys,xs, ze,ye,xe;
	cin >> zs >> ys >> xs >> ze>> ye>> xe;
	  
	image.fill(0);
	for (int z=zs; z<=ze; z++)
	  for (int y=ys; y <= ye; y++)
	    for (int x=xs; x<= xe; x++)
	      image[z][y][x] = 1; 
	  
	display(Tensor3D<float>(image));
	  
        CPUTimer timer;
	timer.restart();
	forward_project(image, segment_pos, segment_neg/*, 
							 segment_pos.min_ring(), segment_pos.max_ring()*/);     

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

	display(Tensor3D<float>(segment_pos), segment_pos.find_max());
	//display(segment_neg);
      }

    }

  return 0;

}


