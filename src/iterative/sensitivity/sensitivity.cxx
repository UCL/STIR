//
// $Id$
//
/*!

  \file
  \ingroup OSMAPOSL

  \brief This is a program to compute the 'sensitivity' image (detection probabilities per voxel). 
  When no input attenuation file is specified, attenuation factors are supposed to be 1, and similar for the normalisation factors.


  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$

  
   \todo use the BinNormalisation class
   \todo put into LogLikBasedAlgorithm

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

/* 
 
  Modification history:
  KT use OSMAPOSLParameters to get most of the parameters to avoid 
     mistakes with unmatched sensitivity images
  KT allow multiplicative normalisation
  KT allow for different sized attenuation image when projectors can handle it
   */


#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/utilities.h"

#include "stir/LogLikBased/common.h"

#include "stir/recon_buildblock/distributable.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"

#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include <typeinfo>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::endl;
#endif


START_NAMESPACE_STIR

class sensparameters{
  public:
 int limit_segments;
  bool zero_seg0_end_planes;
  int rim_truncation_sino;
} globals;



void
 compute_sensitivity_image(DiscretisedDensity<3,float>& result,
                           shared_ptr<ProjData> const& proj_data_ptr,
			   shared_ptr<DiscretisedDensity<3,float> > const& attenuation_image_ptr,
			   const bool do_attenuation,
                           shared_ptr<ProjData> const& atten_proj_data_ptr, 
			   shared_ptr<ProjData> const& norm_proj_data_ptr, 
			   //const Normalisation& normalisation,
			   const sensparameters &globals)
{

    int min_segment = -globals.limit_segments; // KT 30/05/2002 use new convention of distributable_* functions
    int max_segment = globals.limit_segments;

    // TODO attenuation factors
    assert(atten_proj_data_ptr.use_count() == 0);
    distributable_compute_sensitivity_image(result,
					    proj_data_ptr->get_proj_data_info_ptr()->clone(),
                                            do_attenuation ? attenuation_image_ptr.get() : NULL,
					    0,
					    1,
					    min_segment,
					    max_segment,
					    globals.zero_seg0_end_planes,
					    norm_proj_data_ptr);
}


void do_sensitivity(const char * const par_filename)
{
  OSMAPOSLReconstruction parameters;
  if (!parameters.parse(par_filename))
  {
    warning("Error parsing input file %s, exiting\n", par_filename);
    exit(EXIT_FAILURE);
  }
  if (parameters.max_segment_num_to_process==-1)
    parameters.max_segment_num_to_process =
      parameters.proj_data_ptr->get_max_segment_num();

  globals.rim_truncation_sino = rim_truncation_sino;
  string out_filename = parameters.sensitivity_image_filename;
  globals.zero_seg0_end_planes = parameters.zero_seg0_end_planes==1;
  const shared_ptr<ProjData> proj_data_ptr = parameters.proj_data_ptr;
  globals.limit_segments = parameters.max_segment_num_to_process;
  

  // get attenuation info
  bool do_attenuation;
  shared_ptr<DiscretisedDensity<3,float> >
    attenuation_image_ptr = 0;

  {
    char atten_name[max_filename_length];
    ask_filename_with_extension(atten_name, 
				"Get attenuation image from which file (0 = 0's): ",
				"");    
    
    // KT 18/08/2000 compare whole string instead of only first character
    if(strcmp(atten_name,"0")==0)
    {
      do_attenuation = false;
    }
    else
    {
      do_attenuation = true;

     // Read from file      
      attenuation_image_ptr =
        DiscretisedDensity<3,float>::read_from_file(atten_name);
    
      cerr << "WARNING: attenuation image data are supposed to be in units cm^-1\n"
	"Reference: water has mu .096 cm^-1" << endl;
      cerr<< "Max in attenuation image:" 
	  << attenuation_image_ptr->find_max() << endl;
#ifndef NORESCALE
      /*
	cerr << "WARNING: multiplying attenuation image by x-voxel size "
	<< " to correct for scale factor in forward projectors...\n";
      */
      // projectors work in pixel units, so convert attenuation data 
      // from cm^-1 to pixel_units^-1
      const VoxelsOnCartesianGrid<float> * attenuation_image_voxels_ptr =
         dynamic_cast<const VoxelsOnCartesianGrid<float> *>(attenuation_image_ptr.get());
      if (attenuation_image_voxels_ptr == 0)
        error("Can only handle VoxelsOnCartesianGrid for the attenuation image\n");

      const float rescale = 
	attenuation_image_voxels_ptr->get_voxel_size().x()/10;
#else
      // projectors work in mm, so convert attenuation data 
      // from cm^-1 to mm^-1
      const float rescale = 1/10.F;
#endif
      *attenuation_image_ptr *= rescale;      
    }
  }

  shared_ptr<ProjData> atten_proj_data_ptr = 0;

  // get normalisation info
  shared_ptr<ProjData> norm_proj_data_ptr = 0;
  {
    char norm_name[max_filename_length];
    ask_filename_with_extension(norm_name, 
				"Get normalisation factors from which file (1 = 1's):",
				"");    
    
    if(strcmp(norm_name,"1")!=0)
    {
      norm_proj_data_ptr = ProjData::read_from_file(norm_name);
      // check sizes etc.
      shared_ptr<ProjData> data_to_reconstruct_ptr =
        ProjData::read_from_file(parameters.input_filename);
      if (*norm_proj_data_ptr->get_proj_data_info_ptr() !=
          *data_to_reconstruct_ptr->get_proj_data_info_ptr())
          error("Normalisation file and input projection file must have identical characteristics.\n");
    }
  }

  // Initialise the sensitivity image  
  shared_ptr<DiscretisedDensity<3,float> > result_ptr;
  {
    // TODO replace by IterativeReconstruction::get_initial_image_ptr
    if (parameters.initial_image_filename=="1")
      {
	result_ptr =
	  new VoxelsOnCartesianGrid<float> (*parameters.proj_data_ptr->get_proj_data_info_ptr(),
					    static_cast<float>(parameters.zoom),
					    CartesianCoordinate3D<float>(static_cast<float>(parameters.Zoffset),
									 static_cast<float>(parameters.Yoffset),
									 static_cast<float>(parameters.Xoffset)),
					    CartesianCoordinate3D<int>(parameters.output_image_size_z,
								       parameters.output_image_size_xy,
								       parameters.output_image_size_xy));
      }
    else
      {
	result_ptr = 
	  DiscretisedDensity<3,float>::read_from_file(parameters.initial_image_filename);
	result_ptr->fill(0);
      }
  }

  // set_up projectors (ugly!)
  {
    shared_ptr<ForwardProjectorByBin> forward_projector_ptr =
      parameters.projector_pair_ptr->get_forward_projector_sptr();
    shared_ptr<BackProjectorByBin> back_projector_ptr =
      parameters.projector_pair_ptr->get_back_projector_sptr();
    
    if (do_attenuation && dynamic_cast<const ProjectorByBinPairUsingProjMatrixByBin*>(parameters.projector_pair_ptr.get()) != NULL)
    {
      // attenuation image and output image must have identical info
      // TODO somehow remove restriction
      {
        if (typeid(*attenuation_image_ptr) != typeid(*result_ptr))
          error("single projection matrix used: attenuation image and result should be the same type of DiscretisedDensity. Sorry.\n");
        if (attenuation_image_ptr->get_origin() != result_ptr->get_origin())
          error("single projection matrix used: Currently, attenuation and result should have the same origin. Sorry.\n");
        if (attenuation_image_ptr->get_index_range() != result_ptr->get_index_range())
          error("single projection matrix used: Currently, attenuation and result should have the same index ranges. Sorry.\n");
        {
          DiscretisedDensityOnCartesianGrid<3,float> const *attn_ptr =
            dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const *>(attenuation_image_ptr.get());
          if (attn_ptr != 0)
          {
            // we can now check on grid_spacing
            DiscretisedDensityOnCartesianGrid<3,float> const *image_ptr =
              dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const *>(result_ptr.get());
            if (attn_ptr->get_grid_spacing() != image_ptr->get_grid_spacing())
              error("single projection matrix used: Currently, attenuation and result should have the same grid spacing. Sorry.\n");
          }
        }
        // TODO ensure compatible info for any type of DiscretisedDensity
      }
      parameters.projector_pair_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(), 
                                            attenuation_image_ptr);
    }
    else
    {
      shared_ptr<ProjDataInfo> proj_data_info_sptr =
	proj_data_ptr->get_proj_data_info_ptr()->clone();
      if (do_attenuation)
	{
	  // attenuation image and output image can have different info
	  forward_projector_ptr->set_up(proj_data_info_sptr, attenuation_image_ptr);
	}
      back_projector_ptr->set_up(proj_data_info_sptr, result_ptr);
    }

    set_projectors_and_symmetries(forward_projector_ptr, 
				  back_projector_ptr, 
				  back_projector_ptr->get_symmetries_used()->clone());
  } // end set_up projectors

  // Compute the sensitivity image  
  compute_sensitivity_image(*result_ptr,
			    proj_data_ptr, 
			    attenuation_image_ptr,  do_attenuation, 
			    atten_proj_data_ptr,
			    norm_proj_data_ptr,
			    globals);
  parameters.output_file_format_ptr->
      write_to_file(out_filename, *result_ptr);

  cerr << "min and max in image " << result_ptr->find_min() 
       << " " << result_ptr->find_max() << endl;

}
END_NAMESPACE_STIR


USING_NAMESPACE_STIR

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{
  if(argc!=2) 
  {
    cerr<<"Usage: sensitivity OSMAPOSL_par_file\n"
        <<"The par-file will be used to get the scanner, mashing etc. details" 
	<< endl; 
    return (EXIT_FAILURE);
  }

  do_sensitivity(argv[1]);
  return EXIT_SUCCESS;

}
