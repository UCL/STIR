//
// @(#)IterativeReconstructionParameters.h	1.4 00/07/05
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

  \date    00/07/05
  \version 1.4

*/



#include "tomo/ImageProcessor.h"
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


  //! inter-iteration filter
  //shared_ptr<
    ImageProcessor<3,float>* inter_iteration_filter_ptr;

   //! post-filter
  //shared_ptr<
    ImageProcessor<3,float>*  post_filter_ptr;


  
  //! subiteration interval at which to apply inter-iteration filters 
  int inter_iteration_filter_interval;

  virtual void set_defaults();
  virtual void initialise_keymap();




};

END_NAMESPACE_TOMO



#endif // __IterativeReconstructionParameters_h__
