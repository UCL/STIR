//
// $Id$
//

#ifndef __IterativeReconstructionParameters_h__
#define __IterativeReconstructionParameters_h__

/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the IterativeReconstructionParameters class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$
  \version $Revision$

*/



#include "ImageFilter.h"
#include "recon_buildblock/ReconstructionParameters.h"

START_NAMESPACE_TOMO

/*! 

 \brief base parameter class for iterative algorithms

 */

class IterativeReconstructionParameters: public ReconstructionParameters
{

protected:

  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();



public:

  //! constructor
  IterativeReconstructionParameters();

  //! lists the parameter values
  virtual string parameter_info() const;

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();
    

  //TODO make const again

  //! the maximum allowed number of full iterations
  int max_num_full_iterations;

  //! the number of ordered subsets
  int num_subsets;

  //! the number of subiterations 
  int num_subiterations;

  //! value with which to initialize the subiteration counter
  int start_subiteration_num;

  //! name of the file containing the image data for intializing the reconstruction
  string initial_image_filename;

  //! the starting subset number
  int start_subset_num;

  //TODO rename

  //! subiteration interval at which images will be saved
  int save_interval;

  //! signals whether to zero the data in the end planes of the projection data
  int zero_seg0_end_planes;

  //! signals whether to randomise the subset order in each iteration
  int randomise_subset_order;

  //! subiteration interval at which to apply inter-iteration filters 
  int inter_iteration_filter_interval;

  //! inter-iteration filter type name
  string inter_iteration_filter_type;

  //! inter-iteration filter FWHM for transaxial direction filtering
  double inter_iteration_filter_fwhmxy_dir;

  //! inter-iteration filter FWHM for axial direction filtering
  double inter_iteration_filter_fwhmz_dir;

  //! inter-iteration filter Metz power for transaxial direction filtering
  double inter_iteration_filter_Nxy_dir;

  //! inter-iteration filter Metz power for axial direction filtering
  double inter_iteration_filter_Nz_dir;

  //! signals whether or not to apply post-filtering
  int do_post_filtering;

  //! post-filter type name
  string post_filter_type;

  //! post-filter FWHM for transaxial direction filtering
  double post_filter_fwhmxy_dir;

  //! post-filter FWHM for axial direction filtering
  double post_filter_fwhmz_dir;

  //! post-filter Metz power for transaxial direction filtering
  double post_filter_Nxy_dir;

  //! post-filter Metz power for axial direction filtering
  double post_filter_Nz_dir;

  // TODO make shared_ptr

  //! inter-iteration filter object
  ImageFilter inter_iteration_filter;

  //!post-filter object
  ImageFilter post_filter;


};

END_NAMESPACE_TOMO



#endif // __IterativeReconstructionParameters_h__
