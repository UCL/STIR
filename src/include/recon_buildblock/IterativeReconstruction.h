//
// @(#)IterativeReconstruction.h	1.4 00/06/15
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

  \date    00/06/15
  \version 1.4
*/
/* Modification history

   KT 10122001
   - added get_initial_image_ptr and 0 argument reconstruct()
*/
#include "recon_buildblock/Reconstruction.h"
#include "recon_buildblock/IterativeReconstructionParameters.h"

START_NAMESPACE_TOMO

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
  IterativeReconstructionParameters& get_parameters()
    {
      return static_cast<IterativeReconstructionParameters&>(params());
    }

  //! accessor for the external parameters
  const IterativeReconstructionParameters& get_parameters() const
    {
      return static_cast<const IterativeReconstructionParameters&>(params());
    }


  //! the subiteration counter
  int subiteration_num;

  //! used to abort the loop over iterations
  bool terminate_iterations;


 private:
 
  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params()=0;

  //! a workaround for compilers not supporting covariant return types
  virtual const ReconstructionParameters& params() const=0;

};







END_NAMESPACE_TOMO






#endif
// __IterativeReconstruction_h__

