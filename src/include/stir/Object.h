//
// $Id$: $Date$
//
/*!

  \file

  \brief Declaration of class Object

  \author Kris Thielemans
  
  \date $Date$

  \version $Revision$
*/

#include "tomo/common.h"

#include <string>
#ifndef TOMO_NO_NAMESPACE
using std::string;
#endif

#ifndef __Object_H__
#define __Object_H__

START_NAMESPACE_TOMO

/*! \brief Base class for all classes that can parse .par files (and more?) */

class Object
{
public:
  virtual ~Object() {}
  // TODO would like to have the next function const, 
  // but can't because of ClassWithParsing::parameter_info
  virtual string parameter_info()  = 0;
  virtual string get_registered_name() const= 0;
};
END_NAMESPACE_TOMO
#endif

