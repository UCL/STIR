//
// $Id$: $Date$
//
/*!
  \file 
 
  \brief defines the warning() function

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/
#include "Tomography_common.h"

#include <cstdarg>
#include <cstdlib>

START_NAMESPACE_TOMO

void warning(const char *const s, ...)
{
  va_list ap;
  va_start(ap, s);
  vfprintf(stderr, s, ap);
}

END_NAMESPACE_TOMO
