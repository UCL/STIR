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

  \date    $Date$
  \version $Revision$
*/

#include "recon_buildblock/Reconstruction.h"
#include "recon_buildblock/IterativeReconstructionParameters.h"

START_NAMESPACE_TOMO

/*! 
  \brief base class for iterative reconstruction objects
  \ingroup recon_buildblock
 */

class IterativeReconstruction : public Reconstruction
{
 
    
public:

  //! accessor for the subiteration counter
  int get_subiteration_num() const
    {return subiteration_num;}

  //! executes the reconstruction
  virtual Succeeded 
    reconstruct(shared_ptr<DiscretisedDensity<3,float> > const& target_image_ptr);

protected:
 
  IterativeReconstruction();

  //! operations prior to the iterations
  virtual void recon_set_up(shared_ptr <DiscretisedDensity<3,float> > const& target_image_ptr)=0;

  //! the principal operations for updating the image iterates at each iteration
  virtual void update_image_estimate(DiscretisedDensity<3,float> &current_image_estimate)=0;

  //! operations for the end of the iteration
  virtual void end_of_iteration_processing(DiscretisedDensity<3,float> &current_image_estimate)=0;

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

