//
// $Id$
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

#include "distributable.h"
#include "mle_common.h"

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
			   PETImageOfVolume& attenuation_image,
			  const bool do_attenuation,
			  const Normalisation& normalisation,
			  const parameters &globals);

// AZ 06/10/99: added parallel main()
#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
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
  
  //globals.limit_segments++;
  
  // KT 04/11/98 new
  // KT 13/11/98 set defaults to work properly
  globals.zero_seg0_end_planes =
    ask("Zero end planes of segment 0 ?", 
        s3d.get_min_ring_difference(0) == s3d.get_max_ring_difference(0));

  // KT 09/11/98 new
  // KT 13/11/98 hard wire in 
  // AZ 07/10/99 assign to the global one
  globals.rim_truncation_sino = ::rim_truncation_sino;
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
					   PETImageOfVolume& attenuation_image,
					   const bool do_attenuation,
					   const Normalisation& normalisation,
					   const parameters &globals)
{
  PETImageOfVolume 
    result = attenuation_image.get_empty_copy();

#ifdef TEST

  for (int segment_num = 0; segment_num <= globals.limit_segments; segment_num++)
  {
    int min_segment = segment_num;
    int max_segment = segment_num;
    
    result.fill(0);
      
#else

    int min_segment = 0;
    int max_segment = globals.limit_segments;

#endif

    distributable_compute_sensitivity_image(result,
					    s3d,
					    attenuation_image,
					    do_attenuation,
					    0,
					    1,
					    min_segment,
					    max_segment,
					    globals.zero_seg0_end_planes,
					    NULL); //TODO: multiplicative_sinogram goes here

#ifdef TEST

    char fname[20];
    sprintf(fname, "seg_%d.prof", segment_num);
    cerr << "Writing horizontal profiles to " << fname << endl;
    ofstream profile(fname);
    if (!profile)
    {
      cerr << "Couldn't open " << fname << endl;
    }
    else
    {
      for (int z = image_result.get_min_z(); z <= image_result.get_max_z(); z++) 
      { 
	for (int x = image_result.get_min_x(); x <= image_result.get_max_x(); x++)
	  profile<<image_result[z][0][x]<<" ";
	profile << "\n";
      };
    };
  };

#endif

  return result;
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
  
  // KT 22/03/98 removed last arg Real(1) for appropriate defaults
  return new PETSinogramOfVolume(
    scan_info, 
    span, 
    scan_info.get_scanner().type != PETScannerInfo::Advance ?
    scan_info.get_num_rings()-1 : 11,
    *out, 0UL,
    PETSinogramOfVolume::SegmentViewRingBin,
    NumericType::FLOAT);
}


void RPC_process_seg0_view(PETImageOfVolume& lambda, PETImageOfVolume& image_x,
			   PETSegmentBySinogram* y, int view, int view45,
			   int& count, int& count2, float* f /* = NULL */,
			   PETSegmentBySinogram* binwise_correction)
{
  assert(y != NULL || binwise_correction != NULL);
  //  assert(binwise_correction != NULL);

  PETSegmentBySinogram y_bar = (y != NULL) ? *y : *binwise_correction;
  y_bar.fill(0.0);

  if (f != NULL)
  {
    forward_project_2D(lambda, y_bar, view);
    PETViewgram viewgram = y_bar.get_viewgram(view);
    viewgram *= -1;
    in_place_exp(viewgram);
    y_bar.set_viewgram(viewgram);

    // current convention -- in 2D case forward project view45 and view0
    // together to be consistent with 2D backprojector
    if (view == 0)
    {
      forward_project_2D(lambda, y_bar, view45);
      PETViewgram viewgram = y_bar.get_viewgram(view45);
      viewgram *= -1;
      in_place_exp(viewgram);
      y_bar.set_viewgram(viewgram);
    };
  }
  else
  {
    y_bar.fill(1); 
  }

  if (RPC_slave_sens_zero_seg0_end_planes)
  {
    y_bar[y_bar.get_min_ring()].fill(0);
    y_bar[y_bar.get_max_ring()].fill(0);
  };
      
  int view90 = view45 * 2;
  truncate_rim(y_bar, rim_truncation_sino, view);
  truncate_rim(y_bar, rim_truncation_sino, view + view90);
  if (view == 0)
  {
    truncate_rim(y_bar, rim_truncation_sino, view45);
    truncate_rim(y_bar, rim_truncation_sino, view90 + view45);
  }
  else
  {

    truncate_rim(y_bar, rim_truncation_sino, view90 - view);
    truncate_rim(y_bar, rim_truncation_sino, view90 * 2 - view);
  };

  // normalisation.apply(segment);

  if (binwise_correction != NULL)
  {
    y_bar *= *binwise_correction;
  };

  Backprojection_2D(y_bar, image_x, view);
};      

// AZ&MJ 03/10/99 Added bc_*
void RPC_process_4_viewgrams(PETImageOfVolume& lambda, PETImageOfVolume& image_x, 
                             PETViewgram* pos_view, PETViewgram* neg_view,
	                     PETViewgram* pos_plus90, PETViewgram* neg_plus90,
	                     int& count, int& count2, float* f /* = NULL */,
			     PETViewgram* bc_pos_view /* = NULL */, PETViewgram* bc_neg_view /* = NULL */,
			     PETViewgram* bc_pos_plus90 /* = NULL */, PETViewgram* bc_neg_plus90 /* = NULL */)
{
  // AZ 04/10/99: all these should be non-NULL
  //  assert(bc_pos_view != NULL && bc_neg_view != NULL && bc_pos_plus90 != NULL && bc_neg_plus90 != NULL);

  // AZ 04/10/99: these should be either all NULL or all non-NULL
  assert((pos_view != NULL && neg_view != NULL && pos_plus90 != NULL && neg_plus90 != NULL) ||
	 (pos_view == NULL && neg_view == NULL && pos_plus90 == NULL && neg_plus90 == NULL));

  PETViewgram pos_bar_view   = (pos_view != NULL) ? pos_view->get_empty_copy() : bc_pos_view->get_empty_copy();
  PETViewgram neg_bar_view   = (neg_view != NULL) ? neg_view->get_empty_copy() : bc_neg_view->get_empty_copy();
  PETViewgram pos_bar_plus90 = (pos_plus90 != NULL) ? pos_plus90->get_empty_copy() : bc_pos_plus90->get_empty_copy();
  PETViewgram neg_bar_plus90 = (neg_plus90 != NULL) ? neg_plus90->get_empty_copy() : bc_neg_plus90->get_empty_copy();

  if (f != NULL)
  {
    forward_project(lambda,
		    pos_bar_view, 
		    neg_bar_view, 
		    pos_bar_plus90, 
		    neg_bar_plus90);

    pos_bar_view *= -1;
    neg_bar_view *= -1;
    pos_bar_plus90 *= -1;
    neg_bar_plus90 *= -1;

    in_place_exp(pos_bar_view);
    in_place_exp(neg_bar_view);
    in_place_exp(pos_bar_plus90);
    in_place_exp(neg_bar_plus90);
  }
  else
  {
    pos_bar_view.fill(1);
    neg_bar_view.fill(1);
    pos_bar_plus90.fill(1);
    neg_bar_plus90.fill(1);
  };

  truncate_rim(pos_bar_view, rim_truncation_sino);
  truncate_rim(neg_bar_view, rim_truncation_sino);
  truncate_rim(pos_bar_plus90, rim_truncation_sino);
  truncate_rim(neg_bar_plus90, rim_truncation_sino);

  // TODO:normalise

  if (bc_pos_view != NULL)
  {
    pos_bar_view   *= *bc_pos_view;
    neg_bar_view   *= *bc_neg_view;
    pos_bar_plus90 *= *bc_pos_plus90;
    neg_bar_plus90 *= *bc_neg_plus90;
  };
      
  back_project(pos_bar_view, 
	       neg_bar_view, 
	       pos_bar_plus90, 
	       neg_bar_plus90, 
	       image_x);
};

// AZ&MJ 03/10/99 Added bc_*
void RPC_process_8_viewgrams(PETImageOfVolume& lambda, PETImageOfVolume& image_x, 
                             PETViewgram* pos_view,   PETViewgram* neg_view,
	                     PETViewgram* pos_plus90, PETViewgram* neg_plus90,
                             PETViewgram* pos_min180, PETViewgram* neg_min180,
	                     PETViewgram* pos_min90,  PETViewgram* neg_min90,
	                     int& count, int& count2, float* f /* = NULL */,
			     PETViewgram* bc_pos_view   /* = NULL */, PETViewgram* bc_neg_view   /* = NULL */,
			     PETViewgram* bc_pos_plus90 /* = NULL */, PETViewgram* bc_neg_plus90 /* = NULL */,
			     PETViewgram* bc_pos_min180 /* = NULL */, PETViewgram* bc_neg_min180 /* = NULL */,
			     PETViewgram* bc_pos_min90  /* = NULL */, PETViewgram* bc_neg_min90  /* = NULL */)
{
  // AZ 04/10/99: all these should be non-NULL
  //  assert(pos_view == NULL && neg_view == NULL && pos_plus90 == NULL && neg_plus90 == NULL &&
  //	 pos_min180 == NULL && neg_min180 == NULL && pos_min90 == NULL && neg_min90 == NULL);
  //  assert(bc_pos_view != NULL && bc_neg_view != NULL && bc_pos_plus90 != NULL && bc_neg_plus90 != NULL &&
  // bc_pos_min180 != NULL && bc_neg_min180 != NULL && bc_pos_min90 != NULL && bc_neg_min90 != NULL);

  // AZ 04/10/99: these should be either all NULL or all non-NULL
  assert((pos_view != NULL && neg_view != NULL && pos_plus90 != NULL && neg_plus90 != NULL &&
	  pos_min180 != NULL && neg_min180 != NULL && pos_min90 != NULL && neg_min90 != NULL) ||
	 (pos_view == NULL && neg_view == NULL && pos_plus90 == NULL && neg_plus90 == NULL &&
	  pos_min180 == NULL && neg_min180 == NULL && pos_min90 == NULL && neg_min90 == NULL));

  PETViewgram pos_bar_view   = (pos_view != NULL) ? pos_view->get_empty_copy() : bc_pos_view->get_empty_copy();
  PETViewgram neg_bar_view   = (neg_view != NULL) ? neg_view->get_empty_copy() : bc_neg_view->get_empty_copy();
  PETViewgram pos_bar_plus90 = (pos_plus90 != NULL) ? pos_plus90->get_empty_copy() : bc_pos_plus90->get_empty_copy();
  PETViewgram neg_bar_plus90 = (neg_plus90 != NULL) ? neg_plus90->get_empty_copy() : bc_neg_plus90->get_empty_copy();
  PETViewgram pos_bar_min90  = (pos_min90 != NULL) ? pos_min90->get_empty_copy() : bc_pos_min90->get_empty_copy();
  PETViewgram neg_bar_min90  = (neg_min90 != NULL) ? neg_min90->get_empty_copy() : bc_neg_min90->get_empty_copy();
  PETViewgram pos_bar_min180 = (pos_min180 != NULL) ? pos_min180->get_empty_copy() : bc_pos_min180->get_empty_copy();
  PETViewgram neg_bar_min180 = (neg_min180 != NULL) ? neg_min180->get_empty_copy() : bc_neg_min180->get_empty_copy();

  /*
  PETViewgram pos_bar_view   = bc_pos_view->get_empty_copy();
  PETViewgram neg_bar_view   = bc_neg_view->get_empty_copy();
  PETViewgram pos_bar_plus90 = bc_pos_plus90->get_empty_copy();
  PETViewgram neg_bar_plus90 = bc_neg_plus90->get_empty_copy();
  PETViewgram pos_bar_min90  = bc_pos_min90->get_empty_copy();
  PETViewgram neg_bar_min90  = bc_neg_min90->get_empty_copy();
  PETViewgram pos_bar_min180 = bc_pos_min180->get_empty_copy();
  PETViewgram neg_bar_min180 = bc_neg_min180->get_empty_copy();
  */

  if (f != NULL)
  {
    forward_project(lambda,
		    pos_bar_view, 
		    neg_bar_view, 
		    pos_bar_plus90, 
		    neg_bar_plus90, 
		    pos_bar_min180, 
		    neg_bar_min180, 
		    pos_bar_min90, 
		    neg_bar_min90);
      
    pos_bar_view *= -1;
    neg_bar_view *= -1;
    pos_bar_plus90 *= -1;
    neg_bar_plus90 *= -1;
    pos_bar_min180 *= -1;
    neg_bar_min180 *= -1;
    pos_bar_min90 *= -1;
    neg_bar_min90 *= -1;

    in_place_exp(pos_bar_view);
    in_place_exp(neg_bar_view);
    in_place_exp(pos_bar_plus90);
    in_place_exp(neg_bar_plus90);
    in_place_exp(pos_bar_min180);
    in_place_exp(neg_bar_min180);
    in_place_exp(pos_bar_min90);
    in_place_exp(neg_bar_min90);
  }
  else
  {
    pos_bar_view.fill(1);
    neg_bar_view.fill(1);
    pos_bar_plus90.fill(1);
    neg_bar_plus90.fill(1);
    pos_bar_min180.fill(1);
    neg_bar_min180.fill(1);
    pos_bar_min90.fill(1);
    neg_bar_min90.fill(1);
  };

  truncate_rim(pos_bar_view, rim_truncation_sino);
  truncate_rim(neg_bar_view, rim_truncation_sino);
  truncate_rim(pos_bar_plus90, rim_truncation_sino);
  truncate_rim(neg_bar_plus90, rim_truncation_sino);
  truncate_rim(pos_bar_min180, rim_truncation_sino);
  truncate_rim(neg_bar_min180, rim_truncation_sino);
  truncate_rim(pos_bar_min90, rim_truncation_sino);
  truncate_rim(neg_bar_min90, rim_truncation_sino);

  // TODO:normalise

  if (bc_pos_view != NULL)
  {
    pos_bar_view   *= *bc_pos_view;
    neg_bar_view   *= *bc_neg_view;
    pos_bar_plus90 *= *bc_pos_plus90;
    neg_bar_plus90 *= *bc_neg_plus90;
    pos_bar_min90  *= *bc_pos_min90;
    neg_bar_min90  *= *bc_neg_min90;
    pos_bar_min180 *= *bc_pos_min180;
    neg_bar_min180 *= *bc_neg_min180;
  };
      
  back_project(pos_bar_view, 
	       neg_bar_view, 
	       pos_bar_plus90, 
	       neg_bar_plus90, 
	       pos_bar_min180, 
	       neg_bar_min180, 
	       pos_bar_min90, 
	       neg_bar_min90, 
	       image_x);
};
