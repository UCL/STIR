// @(#)Reconstruction.h	1.4 00/06/15
#ifndef __Reconstruction_H__
#define __Reconstruction_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the Reconstruction class

  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project

  \date    00/06/15
  \version 1.4
*/


#include "TimedObject.h"
#include <string>


#ifndef TOMO_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_TOMO

class ReconstructionParameters;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename T> class shared_ptr;

/*!
 \brief base class for all Reconstructions

  For convenience, the class is derived from TimedObject. It is the 
  responsibility of the derived class to run these timers though.
*/

class Reconstruction : public TimedObject 
{
public:
  //! virtual destructor
  virtual ~Reconstruction() {};
  
  //! gives method information
  virtual string method_info() const = 0;
  
  //! lists the parameters
  virtual string parameter_info()  = 0;
  
  //! executes the reconstruction
  /*!
    \param target_image_ptr The result of the reconstruction is stored in *target_image_ptr.
    For iterative reconstructions, *target_image_ptr is used as an initial estimate.
    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    reconstruct(shared_ptr<DiscretisedDensity<3,float> > const& target_image_ptr) = 0;

  
  //! accessor for the external parameters
  ReconstructionParameters& get_parameters()
    {
      return static_cast<ReconstructionParameters&>(params());
    }

  //! accessor for the external parameters
  const ReconstructionParameters& get_parameters() const
    {
      return static_cast<const ReconstructionParameters&>(params());
    }


private:

  //! a workaround for compilers not supporting covariant return types
  virtual ReconstructionParameters& params()=0;

  //! a workaround for compilers not supporting covariant return types
  virtual const ReconstructionParameters& params() const=0;


};

END_NAMESPACE_TOMO

    
#endif

// __Reconstruction_H__
