//
// $Id$
//
#ifndef __IterativeReconstruction_h__
#define __IterativeReconstruction_h__

/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the IterativeReconstruction class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Version:$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
/* Modification history

   KT 10122001
   - added get_initial_image_ptr and 0 argument reconstruct()
*/
#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/shared_ptr.h"
#include "stir/ImageProcessor.h"


START_NAMESPACE_STIR

/*! 
  \brief base class for iterative reconstruction objects
  \ingroup recon_buildblock

  \todo move subset things somewhere else
 */

class IterativeReconstruction : public Reconstruction
{
 
    
public:

  //! accessor for the subiteration counter
  int get_subiteration_num() const
    {return subiteration_num;}

  //! Gets a pointer to the initial image
  /*! This is either read from file, or constructed by construct_target_image_ptr(). 
      In the latter case, its values are set to 0 or 1, depending on the value
      of IterativeReconstructionParameters::initial_image_filename.

      \todo Dependency on explicit strings "1" or "0" in 
      IterativeReconstructionParameters::initial_image_filename is not nice. 
  */
  virtual DiscretisedDensity<3,float> *
    get_initial_image_ptr() const; // KT 10122001 new

  //! executes the reconstruction
  /*!
    Calls get_initial_image_ptr() and then 1 argument reconstruct().
    See end_of_iteration_processing() for info on saving to file.

    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    reconstruct();  // KT 10122001 new

  //! executes the reconstruction with \a target_image_ptr as initial value
  /*! After calling recon_set_up(), repeatedly calls update_image_estimate(); end_of_iteration_processing();
      See end_of_iteration_processing() for info on saving to file.
  */
  virtual Succeeded 
    reconstruct(shared_ptr<DiscretisedDensity<3,float> > const& target_image_ptr);

protected:
 
  IterativeReconstruction();

  //! operations prior to the iterations
  virtual void recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr)=0;

  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate)=0;

  //! operations for the end of the iteration
  /*! At specific subiteration numbers, this 
      <ul>
      <li>applies the inter-filtering and/or post-filtering image processor,
      <li>writes the current image to file at the designated subiteration numbers 
      (including the final one). Filenames used are determined by
      ReconstructionParameters::output_filename_prefix
      </ul>
      If your derived class redefines this virtual function, you will
      probably want to call 
      IterativeReconstruction::end_of_iteration_processing() in there anyway.
  */
  // KT 14/12/2001 remove =0 as it's not a pure virtual and the default implementation is usually fine.
  virtual void end_of_iteration_processing(DiscretisedDensity<3,float> &current_image_estimate);

  //! used to randomly generate a subset sequence order for the current iteration
  VectorWithOffset<int> randomly_permute_subset_order();

  //! accessor for the external parameters
  IterativeReconstruction& get_parameters()
    {
      return *this;
    }

  //! accessor for the external parameters
  const IterativeReconstruction& get_parameters() const
    {
      return *this;
    }


  //! the subiteration counter
  int subiteration_num;

  //! used to abort the loop over iterations
  bool terminate_iterations;


  // parameters
 protected:
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
  shared_ptr<ImageProcessor<3,float> > inter_iteration_filter_ptr;

  //! post-filter
  shared_ptr<ImageProcessor<3,float> >  post_filter_ptr;


  
  //! subiteration interval at which to apply inter-iteration filters 
  int inter_iteration_filter_interval;

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

  virtual void set_defaults();
  virtual void initialise_keymap();
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();


};

END_NAMESPACE_STIR

#endif
// __IterativeReconstruction_h__

