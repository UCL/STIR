//
// $Id$
//
#ifndef __stir_recon_buildblock_Reconstruction_H__
#define __stir_recon_buildblock_Reconstruction_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the Reconstruction class

  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
/* Modification history

   KT 10122001
   - added construct_target_image_ptr and 0 argument reconstruct()
*/


#include "stir/TimedObject.h"
#include <string>


#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

class ReconstructionParameters;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename T> class shared_ptr;
class Succeeded;

/*!
  \brief base class for all Reconstructions
  \ingroup recon_buildblock
  As there is not a lot of commonality between different reconstruction algorithms,
  this base class is rather basic. It essentially takes care of constructing a target
  image, calls the virtual reconstruct() function, and writes the result to file.

  For convenience, the class is derived from TimedObject. It is the 
  responsibility of the derived class to run these timers though.

  \todo merge ReconstructionParameters hierarchy with this one.
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
  

  //! Creates a suitable target_image as determined by the parameters
  virtual DiscretisedDensity<3,float>* 
    construct_target_image_ptr() const; // KT 10122001 new
  
  //! executes the reconstruction
  /*!
    Calls construct_target_image_ptr() and then 1 argument reconstruct().
    At the end of the reconstruction, the final image is saved to file as given in 
    ReconstructionParameters::output_filename_prefix. 

    This behaviour can be modified by a derived class (for instance see 
    IterativeReconstruction::reconstruct()).

    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    reconstruct(); // KT 10122001 new

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

END_NAMESPACE_STIR

    
#endif

