//
// $Id$: $Date$
//
#ifndef __IterativeReconstruction_h__
#define __IterativeReconstruction_h__

/*!
  \file 
  \ingroup reconstruction
 
  \brief declares the IterativeReconstruction class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/

#include "recon_buildblock/Reconstruction.h"
#include "recon_buildblock/IterativeReconstructionParameters.h"

START_NAMESPACE_TOMO

/*! \brief base class for iterative reconstruction objects

 */


class IterativeReconstruction : public Reconstruction
{
 
    
 public:

  //! accessor for the subiteration counter
  int get_subiteration_num() const
    {return subiteration_num;}

  //! executes the reconstruction
  virtual void reconstruct(PETImageOfVolume &target_image);

 protected:


  //! operations prior to the iterations
  virtual void recon_set_up(PETImageOfVolume &target_image)=0;

  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(PETImageOfVolume &current_image_estimate)=0;

  //! operations for the end of the iteration
  virtual void end_of_iteration_processing(PETImageOfVolume &current_image_estimate)=0;

  //MJ for appendability

  //! operations prior to the iterations common to all iterative algorithms
  void iterative_common_recon_set_up(PETImageOfVolume &target_image);

  //! operations for the end of the iteration common to all iterative algorithms
  void iterative_common_end_of_iteration_processing(PETImageOfVolume &current_image_estimate);

  //! used to randomly generate a subset sequence order for the current iteration
  VectorWithOffset<int> randomly_permute_subset_order();

  //MJ seemingly dumb, but params() may eventually have to return a Reconstruction object


  //! accessor for the external parameters
  IterativeReconstructionParameters& get_parameters()
    {
      return static_cast<IterativeReconstructionParameters&>(params());
    }

  /*
  const IterativeReconstructionParameters& get_parameters() const
    {
      return static_cast<const IterativeReconstructionParameters&>(params());
    }
    */

  
  // Ultimately the argument will be a PETStudy or the like
  
  //! customized constructor that can call pure virtual functions
  void IterativeReconstruction_ctor(char* parameter_filename="");

  //! customized destructor that can call pure virtual functions
  void IterativeReconstruction_dtor();

  //! the subiteration counter
  int subiteration_num;

  //! used to abort the loop over iterations
  bool terminate_iterations;


 private:
 
  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params()=0;


};







END_NAMESPACE_TOMO






#endif
// __IterativeReconstruction_h__

