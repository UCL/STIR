//
// $Id$: $Date$
//


/* This is a program to compute the 'sensitivity' image (detection probabilities per voxel). When no input attenuation file is specified, the result is just the backprojection of all 1's.

   if TEST is defined, results of different ring differences (i.e. the profiles   through the centres of all planes) are
   stored as separate files (seg_x.dat) and the sensitivity image is
   not computed.
   if TEST is not defined, output is written to a user-specified output file

   By Matthew Jacobson and Kris Thielemans

   //KT&MJ 11/08/98 introduce do_attenuation
   TODO:
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
			  const bool do_attenuation,
			  const Normalisation& normalisation);

int main(int argc, char *argv[])
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
      // KT&MJ 11/08/98 allow for higher sampling in z

      //scanner.num_bins /= 4; 
      //scanner.num_views /= 4; 
      cerr << "Warning, using 3 times more rings";
      scanner.num_rings *= 3;
  
    
      scanner.bin_size = 2* scanner.FOV_radius / scanner.num_bins;
      scanner.ring_spacing = scanner.FOV_axial / scanner.num_rings;
    }

  // KT 14/08/98 added conditional
#ifndef TEST
  char out_filename[200];
  cout << endl << "Output to which file ?";
  cin >> out_filename;
#endif


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

  bool do_attenuation;
  PETImageOfVolume 
    attenuation_image(Tensor3D<float>(
				      0, 2*scanner.num_rings-2,
				      (-scanner.num_bins/2), max_bin,
				      (-scanner.num_bins/2), max_bin),
		      origin,
		      voxel_size);

  {
    char atten_name[100];
    
    // if(batch) strcpy(filename2,argv[4]);
    //else{
    cout<<endl;
    
    cout << endl << "Get attenuation image from which file (0 = 0's): ";
    cin >> atten_name;
    //}
    
    if(atten_name[0]=='0')
    {
      do_attenuation = false;
    }
    else
    {
      do_attenuation = true;

      ifstream atten_img;
      open_read_binary(atten_img, atten_name);
      attenuation_image.read_data(atten_img);   
    }
  }




  // Compute the sensitivity image  
  PETImageOfVolume result =
    compute_sensitivity_image(scanner, attenuation_image,  do_attenuation, Normalisation ());
#ifndef TEST
  result+=(float)ZERO_TOL;

  // Write it to file
  ofstream sensitivity_data;
  open_write_binary(sensitivity_data, out_filename);
  result.write_data(sensitivity_data);

  cerr << "min and max in image " << result.find_min() 
       << " " << result.find_max() << endl;
  display(Tensor3D<float> (result), result.find_max());
#endif

  return 0;

}


PETImageOfVolume compute_sensitivity_image(const PETScannerInfo& scanner,
					   const PETImageOfVolume& attenuation_image,
					   const bool do_attenuation,
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
  { cerr<<endl<<"Processing segment 0"<<endl;

    PETSegmentBySinogram
      segment(
	      Tensor3D<float>(0, scanner.num_rings-1, 
			      0, scanner.num_views-1,
			      -scanner.num_bins/2, max_bin),
	      &scanner,
	      0);

    
    if (do_attenuation)
    {
      //KT&MJ 11/08/98 use 2D forward projector
      
      cerr<<"Starting forward project"<<endl;

      forward_project_2D(attenuation_image, segment);	  

      cerr<<"Finished forward project"<<endl;
      //display(Tensor3D<float>(segment), segment.find_max());
      
      segment *= -1;
      in_place_exp(segment);
    }
    else
    {
      segment.fill(1); 
    }
      
    normalisation.apply(segment);


    cerr<<endl<<"Starting backproject"<<endl;
    Backprojection_2D(segment, image_result);
    cerr<<endl<<"Finished backproject"<<endl;

    /* cerr << "min and max in image " << image_result.find_min() 
       << " " << image_result.find_max() << endl;
       display(Tensor3D<float> (image_result), image_result.find_max());
    */
#ifdef TEST
    {
      // KT&MJ 12/08/98 output only profiles
      char fname[20];
/*
      sprintf(fname, "seg_%d.dat", 0);
      // Write it to file
      ofstream segment_data;
      open_write_binary(segment_data, fname);
      image_result.write_data(segment_data);
*/
      sprintf(fname, "seg_%d.prof", 0);
      cerr << "Writing horizontal profiles to " << fname << endl;
      ofstream profile(fname);
      if (!profile)
      { cerr << "Couldn't open " << fname; }

      for (int z=image_result.get_min_z(); z<= image_result.get_max_z(); z++) 
      { 
	for (int x=image_result.get_min_x(); x<= image_result.get_max_x(); x++)
          profile<<image_result[z][0][x]<<" ";
        profile << "\n";
      }
    }
#endif

  }


  // now do a loop over the other segments
  // now doing all segments

  for (int segment_num = 1; segment_num < scanner.num_rings ; segment_num++){

    cerr<<endl<<"Processing segment #"<<segment_num<<endl;

#ifdef TEST
    image_result.fill(0);
#endif 

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


    if (do_attenuation)
    {

      cerr<<"Starting forward project"<<endl;
      forward_project(attenuation_image, segment_pos, segment_neg);	       
      cerr<<"Finished forward project"<<endl;
      // display(Tensor3D<float>(segment_pos), segment_pos.find_max());
      
      segment_pos *= -1;
      segment_neg *= -1;
      in_place_exp(segment_pos);
      in_place_exp(segment_neg);
      
    }
    else
    {
      segment_pos.fill(1);
      segment_neg.fill(1);
    }

    normalisation.apply(segment_pos);
    normalisation.apply(segment_neg);

    //KT TODO use by view versions (but also for forward projection)

    cerr<<"Starting backproject"<<endl;
    back_project(segment_pos, segment_neg, image_result);
    cerr<<"Finished backproject"<<endl;

    /*const int nviews = segment_pos.get_num_views();
    const int view90 = nviews / 2;
    for (int view=0; view < segment_pos.get_num_views()/2; view++)
      back_project(segment_pos.get_viewgram(view), 
		 segment_neg.get_viewgram(view),
		 segment_pos.get_viewgram(view90 + view),
		 segment_neg.get_viewgram(view90 + view),
		 image_result);
    */
      
    
    /*
      cerr << "min and max in image " << image_result.find_min() 
      << " " << image_result.find_max() << endl;
      display(Tensor3D<float> (image_result), image_result.find_max());
    */
#ifdef TEST

    {
      char fname[20];
      // KT&MJ 12/08/98 write profiles only
      /*
      sprintf(fname, "seg_%d.dat", segment_num);
      // Write it to file
      ofstream segment_data;
      open_write_binary(segment_data, fname);
      image_result.write_data(segment_data);

      */
      sprintf(fname, "seg_%d.prof", segment_num);
      cerr << "Writing horizontal profiles to " << fname << endl;
      ofstream profile(fname);
      if (!profile)
      { cerr << "Couldn't open " << fname; }

      for (int z=image_result.get_min_z(); z<= image_result.get_max_z(); z++) 
      { 
	for (int x=image_result.get_min_x(); x<= image_result.get_max_x(); x++)
          profile<<image_result[z][0][x]<<" ";
        profile << "\n";
      }
    }
#endif
     
  }
   
  // image_result/=float(scanner.num_views);
 
  return image_result;
}


