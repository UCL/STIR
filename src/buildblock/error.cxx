//
// $Id$
//
/*!
  \file 
 
  \brief defines the error() function

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$

*/
#include "tomo/common.h"

#include <cstdarg>
#include <stdlib.h>

START_NAMESPACE_TOMO

void error(const char *const s, ...)
{
  va_list ap;
  va_start(ap, s);
  vfprintf(stderr, s, ap);
  abort();
}
END_NAMESPACE_TOMO
