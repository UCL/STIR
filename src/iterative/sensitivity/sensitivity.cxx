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


#include "pet_common.h"
// KT 04/11/98 2 new
#include "utilities.h"
#include "interfile.h"

#include "sinodata.h"
#include "imagedata.h"


#include "TensorFunction.h"
//MJ 17/11/98 new
#include "recon_array_functions.h"

#include "recon_buildblock/bckproj.h"
#include "recon_buildblock/fwdproj.h"


#include "display.h"

// KT 13/11/98 new
const int hard_wired_rim_truncation_sino = 4;

class parameters{
  public:
 int limit_segments, MAP_mode, num_subsets, view45, phase_offset, iteration_num;   
  // KT 04/11/98 new
  bool zero_seg0_end_planes;
  // KT 09/11/98 new
  int rim_truncation_sino;
} globals;



// A do-nothing class for normalisation
class Normalisation
{
public:
  virtual void apply(PETSegment&) const {}
};

// KT 01/12/98 function which constructs PSOV (nothing else for the moment)
PETSinogramOfVolume * ask_parameters();
// KT 09/11/98 use PSOV for construction of segments, added const for globals
PETImageOfVolume
compute_sensitivity_image(const PETSinogramOfVolume& s3d,
			  const PETImageOfVolume& attenuation_image,
			  const bool do_attenuation,
			  const Normalisation& normalisation,
			  const parameters &globals);

int main(int argc, char *argv[])
{
  // KT 01/12/98 unsatisfying trick to keep the old way of input
  // but allow a new way using an Interfile header.
  PETSinogramOfVolume *s3d_ptr = 0;

  if(argc!=2) 
  {
    cerr<<"Usage: sensitivity [PSOV-file]\n"
        <<"The PSOV-file will be used to get the scanner, mashing etc. details" 
	<< endl; 
  }
  if (argc>2)
    exit(1);
  
  if(argc==2)
  {
    s3d_ptr = new PETSinogramOfVolume(read_interfile_PSOV(argv[1]));
  }
  else
  {
    s3d_ptr = ask_parameters();
  }

  PETSinogramOfVolume& s3d = *s3d_ptr;

  // KT 01/12/98 from here no changes (except 1...)



  globals.limit_segments=ask_num("Maximum absolute segment number to process: ", 
    0, s3d.get_max_segment(), s3d.get_max_segment() );
  
  globals.limit_segments++;
  
  // KT 04/11/98 new
  // KT 13/11/98 set defaults to work properly
  globals.zero_seg0_end_planes =
    ask("Zero end planes of segment 0 ?", 
        s3d.get_min_ring_difference(0) == s3d.get_max_ring_difference(0));

  // KT 09/11/98 new
  // KT 13/11/98 hard wire in 
  globals.rim_truncation_sino = hard_wired_rim_truncation_sino;
    // ask_num("Number of bins to set to zero ?",0, s3d.get_max_bin(), 4);

  // KT 14/08/98 added conditional
#ifndef TEST
 // KT 04/11/98 nicer
  char out_filename[max_filename_length];
  ask_filename_with_extension(out_filename,
			      "Output to which file (without extension)?",
			      "");
#endif


  bool do_attenuation;
  // KT 09/11/98 use new constructor
  // KT 01/12/98 use the s3d member
  PETImageOfVolume attenuation_image(s3d.scan_info);

  {
    // KT 13/11/98 use ask_
    char atten_name[max_filename_length];
    // KT 10/02/99 tell it reads interfile
    ask_filename_with_extension(atten_name, 
				"Get attenuation image from which file (0 = 0's): (use .hv if you can)",
				"");    
    
    if(atten_name[0]=='0')
    {
      do_attenuation = false;
    }
    else
    {
      do_attenuation = true;

      // KT 13/11/98 read from interfile
      // Read from file by adding to the attenuation_image 
      // (which is 0 at this point)
      // This in principle should allow us to read an 'even-sized' image
      // as += just adds the appropriate ranges
      attenuation_image += read_interfile_image(atten_name);
#if RESCALE
      // KT 10/02/99 temporary plug
      cerr << "WARNING: multiplying by binsize to correct for scale factor in \
forward projectors..." << endl;
      cerr<< "Max before " <<attenuation_image.find_max();
      attenuation_image *= s3d.scan_info.get_bin_size();
      cerr<< ", after " <<attenuation_image.find_max() << endl;
#endif
      /*
      ifstream atten_img;
      open_read_binary(atten_img, atten_name);
      attenuation_image.read_data(atten_img);   
      */
    }
  }




  // Compute the sensitivity image  
  PETImageOfVolume result =
    compute_sensitivity_image(s3d, attenuation_image,  do_attenuation, Normalisation (), globals);
#ifndef TEST
  // KT 04/11/98 removed on request by MJ
  //result+=(float)ZERO_TOL;

  // Write it to file
  // KT 04/11/98 use interfile output
  /*
   ofstream sensitivity_data;
   open_write_binary(sensitivity_data, out_filename);
   result.write_data(sensitivity_data);
   */
  // KT 12/11/98 always write as float
  write_basic_interfile(out_filename, result, NumericType::FLOAT);

  cerr << "min and max in image " << result.find_min() 
       << " " << result.find_max() << endl;

  // KT 04/11/98 disabled
  // display(Tensor3D<float> (result), result.find_max());
#endif

  return 0;

}


PETImageOfVolume compute_sensitivity_image(const PETSinogramOfVolume& s3d,
					   const PETImageOfVolume& attenuation_image,
					   const bool do_attenuation,
					   const Normalisation& normalisation,
					   const parameters &globals)
{
  /* KT 09/11/98 not necessary anymore
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;
  */

  // KT 09/11/98 use new member
  PETImageOfVolume 
    image_result = attenuation_image.get_empty_copy();

  // first do segment 0
  { 
    cerr<<endl<<"Processing segment 0"<<endl;
    
    /* KT 09/11/98 use PSOV
    PETSegmentBySinogram
    segment(
    Tensor3D<float>(0, scanner.num_rings-1, 
    0, scanner.num_views-1,
    -scanner.num_bins/2, max_bin),
    // KT 05/11/98 removed & for PETScanInfo parameter
    scanner,
    0);
    */

    PETSegmentBySinogram
      segment = s3d.get_empty_segment_sino_copy(0);

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
      
    // KT 09/11/98 new
    truncate_rim(segment, globals.rim_truncation_sino);

    normalisation.apply(segment);

    // KT 04/11/98 new
    if (globals.zero_seg0_end_planes)
      {
	cerr << "\nZeroing end-planes of segment 0" << endl;
	segment[segment.get_min_ring()].fill(0);
	segment[segment.get_max_ring()].fill(0);
      }


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

  for (int segment_num = 1; segment_num < globals.limit_segments ; segment_num++)
  {

    cerr<<endl<<"Processing segment #"<<segment_num<<endl;

#ifdef TEST
    image_result.fill(0);
#endif 
    /* KT 09/11/98 use PSOV
    PETSegmentByView 
      segment_pos(
		  Tensor3D<float>(0, scanner.num_views-1, 
				  0, scanner.num_rings-1 - segment_num,
				  -scanner.num_bins/2, max_bin),
	          // KT 05/11/98 removed & for PETScanInfo parameter
	          scanner,
		  segment_num);
    PETSegmentByView 
      segment_neg(
		  Tensor3D<float>(0, scanner.num_views-1, 
				  0, scanner.num_rings-1 - segment_num,
				  -scanner.num_bins/2, max_bin),
		  // KT 05/11/98 removed & for PETScanInfo parameter
		  scanner,
		  -segment_num);
     */
    PETSegmentByView 
      segment_pos = s3d.get_empty_segment_view_copy(segment_num);
    PETSegmentByView 
      segment_neg = s3d.get_empty_segment_view_copy(-segment_num);

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

    // KT 09/11/98 new
    truncate_rim(segment_pos, globals.rim_truncation_sino);
    truncate_rim(segment_neg, globals.rim_truncation_sino);

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
    
  return image_result;
}


PETSinogramOfVolume * 
ask_parameters()
{
    // KT 05/11/98 use scan_info
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

  // KT 09/11/98 allow span
  int span = 1;
  if (scan_info.get_scanner().type != PETScannerInfo::Advance)
    span = ask_num("Span value", 1, scan_info.get_num_rings()/2, 1);

  // KT 09/11/98 use PETSinogramOfVolume
  fstream *out = 0;
  
  return new PETSinogramOfVolume(
    scan_info, 
    span, 
    scan_info.get_scanner().type != PETScannerInfo::Advance ?
    scan_info.get_num_rings()-1 : 11,
    *out, 0UL,
    PETSinogramOfVolume::SegmentViewRingBin,
    NumericType::FLOAT,
    Real(1));
}
