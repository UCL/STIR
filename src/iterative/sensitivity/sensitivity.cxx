//
// $Id$: $Date$
//

/* This is a preliminary program to compute the 'sensitivity' image
   (backprojection of all 1's).
   if ATTENUATION is #defined, it will read an attenuation file
   atten.dat, and take that into account.
   if TEST is defined, results of different ring differences are
   stored as separated files (seg_x.dat) and the sensitivity image is
   not computed.
   if TEST is not defined, output is written to sensitivity.dat

   By Matthew Jacobson and Kris Thielemans

   TODO:
   - find good 'per segment' normalisation. Currently the resulting
     image has ring artefacts.
   - implement the Normalisation class
   */

#include <iostream.h>
#include <fstream.h>
#include "pet_common.h"
#include "PETScannerInfo.h"
#include "sinodata.h"
#include "imagedata.h"


#include "TensorFunction.h"

#include "recon_buildblock/bckproj.h"
#include "recon_buildblock/fwdproj.h"


#include "display.h"

#define ZERO_TOL 0.000001
#define ROOF 40000000.0

// A do-nothing class for normalisation
class Normalisation
{
public:
  virtual void apply(PETSegment&) const {}
};

PETImageOfVolume
compute_sensitivity_image(const PETScannerInfo& scanner,
			  const PETImageOfVolume& attenuation_image,
			  const Normalisation& normalisation);

void 
main(int argc, char *argv[])
{

  if(argc>2) {
    cout<<"Usage: sensitivity [c]";
    exit(1);
  }


  int scanner_num = 0;
  PETScannerInfo scanner;
  char scanner_char[200];


  scanner_num = ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE) ? ", 0,3,0);
  switch( scanner_num )
    {
    case 0:
      scanner = (PETScannerInfo::RPT);
      sprintf(scanner_char,"%s","RPT");
      break;
    case 1:
      scanner = (PETScannerInfo::E953);
      sprintf(scanner_char,"%s","953");   
      break;
    case 2:
      scanner = (PETScannerInfo::E966);
      sprintf(scanner_char,"%s","966"); 
      break;
    case 3:
      scanner = (PETScannerInfo::Advance);
      sprintf(scanner_char,"%s","Advance");   
      break;
    default:
      PETerror("Wrong scanner number\n"); Abort();
    }


 
  if(argc==2)
    {
      scanner.num_bins /= 4; 
      scanner.num_views /= 4; 
      scanner.num_rings /= 1;
  
    
      scanner.bin_size = 2* scanner.FOV_radius / scanner.num_bins;
      scanner.ring_spacing = scanner.FOV_axial / scanner.num_rings;
    }

  Point3D origin(0,0,0);


  Point3D voxel_size(scanner.bin_size,
                     scanner.bin_size,
                     scanner.ring_spacing/2); 





  // Create the image
  // two  manipulations are needed now: 
  // -it needs to have an odd number of elements
  // - it needs to be larger than expected because of some overflow in the projectors
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;

  PETImageOfVolume 
    attenuation_image(Tensor3D<float>(
				      0, 2*scanner.num_rings-2,
				      (-scanner.num_bins/2), max_bin,
				      (-scanner.num_bins/2), max_bin),
		      origin,
		      voxel_size);

#ifdef ATTENUATION

  // Open file with data (should be in floats and with the same sizes as above)
  { 
    ifstream attenuation_data;
    open_read_binary(attenuation_data, "atten.dat");
    attenuation_image.read_data(attenuation_data);
  

    /* some code to read different data type
       {
       Real scale = Real(1);
       attenuation_data.read_data(sino_stream, NumericType::SHORT, scale);
       assert(scale == 1);
       }
    */
  }
#endif

  // Compute the sensitivity image  
  PETImageOfVolume result =
    compute_sensitivity_image(scanner, attenuation_image,  Normalisation ());
#ifndef TEST
  result+=(float)ZERO_TOL;

  // Write it to file
  ofstream sensitivity_data;
  open_write_binary(sensitivity_data, "sensitivity.dat");
  result.write_data(sensitivity_data);

  cerr << "min and max in image " << result.find_min() 
       << " " << result.find_max() << endl;
  display(Tensor3D<float> (result), result.find_max());
#endif

return 0;

}


PETImageOfVolume compute_sensitivity_image(const PETScannerInfo& scanner,
					   const PETImageOfVolume& attenuation_image,
					   const Normalisation& normalisation)
{
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;

  PETImageOfVolume 
    image_result(Tensor3D<float>(0, 2*scanner.num_rings-2,
				 (-scanner.num_bins/2), max_bin,
				 (-scanner.num_bins/2), max_bin),
		 attenuation_image.get_origin(),
		 attenuation_image.get_voxel_size());

  // first do segment 0
  {
    PETSegmentBySinogram
      segment(
	      Tensor3D<float>(0, scanner.num_rings-1, 
			      0, scanner.num_views-1,
			      -scanner.num_bins/2, max_bin),
	      &scanner,
	      0);

      
#ifdef ATTENUATION
    forward_project(attenuation_image, segment, segment, 
		    segment.min_ring(), segment.max_ring());	  
    segment /= 2;

    //display(Tensor3D<float>(segment), segment.find_max());

    segment *= -1;
    in_place_exp(segment);


 
      
    normalisation.apply(segment);
#else
    segment.fill(1);
#endif

    Backprojection_2D(segment, image_result);
      
    /* cerr << "min and max in image " << image_result.find_min() 
       << " " << image_result.find_max() << endl;
       display(Tensor3D<float> (image_result), image_result.find_max());
    */
#ifdef TEST
    {
      char fname[20];
      sprintf(fname, "seg_%d.dat", 0);
    
      // Write it to file
      ofstream segment_data;
      open_write_binary(segment_data, fname);
      image_result.write_data(segment_data);
    }
#endif



  }

  // int num_processed_segments = 1; 
  
  // now do a loop over the other segments
  // do not use last segment, as I think backprojector needs 2 sinograms

  for (int segment_num = 1; segment_num < scanner.num_rings ; segment_num++){

#ifdef TEST
    image_result.fill(0);
#endif 
    //     num_processed_segments += 2; // + 2 as we do a positive and negative segment here


    PETSegmentByView 
      segment_pos(
		  Tensor3D<float>(0, scanner.num_views-1, 
				  0, scanner.num_rings-1 - segment_num,
				  -scanner.num_bins/2, max_bin),
		  &scanner,
		  segment_num);
    PETSegmentByView 
      segment_neg(
		  Tensor3D<float>(0, scanner.num_views-1, 
				  0, scanner.num_rings-1 - segment_num,
				  -scanner.num_bins/2, max_bin),
		  &scanner,
		  -segment_num);


#ifdef ATTENUATION

    forward_project(attenuation_image, segment_pos, segment_neg, 
		    segment_pos.min_ring(), segment_pos.max_ring());	  

     

    // display(Tensor3D<float>(segment_pos), segment_pos.find_max());
      
    segment_pos *= -1;
    segment_neg *= -1;
    in_place_exp(segment_pos);
    in_place_exp(segment_neg);
      
    normalisation.apply(segment_pos);
    normalisation.apply(segment_neg);
#else

    segment_pos.fill(1);
    segment_neg.fill(1);
#endif

    back_project(segment_pos, segment_neg, image_result);
      
    
    /*
      cerr << "min and max in image " << image_result.find_min() 
      << " " << image_result.find_max() << endl;
      display(Tensor3D<float> (image_result), image_result.find_max());
    */
#ifdef TEST

    {
      char fname[20];
      sprintf(fname, "seg_%d.dat", segment_num);
    
      // Write it to file
      ofstream segment_data;
      open_write_binary(segment_data, fname);
      image_result.write_data(segment_data);
    }
#endif
     
  }
   
  // image_result/=float(num_processed_segments * scanner.num_views);
 




  return image_result;
}


