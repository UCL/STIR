// $Id$: $Date$
#ifndef __Reconstruction_H__
#define __Reconstruction_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the Reconstruction class

  \author Kris Thielemans
  \author Claire Labbe
  \author Matthew Jacobson
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/

#include "imagedata.h"
#include "TimedObject.h"
#include <string>
#include "interfile.h"
#include "recon_array_functions.h"
#include "ImageFilter.h"
#include "zoom.h"

#ifndef TOMO_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_TOMO
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
  virtual string parameter_info() const = 0;
  
  //! executes the reconstruction
  virtual void reconstruct(PETImageOfVolume &v) = 0;
  
};

END_NAMESPACE_TOMO

    
#endif

// __Reconstruction_H__
