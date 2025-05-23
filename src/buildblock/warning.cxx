//
//
/*!
  \file 
 
  \brief defines the stir::warning() function

  \author Kris Thielemans
  \author PARAPET project



*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2010, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#include "stir/warning.h"

#include <cstdarg>
#include <iostream>
#include <sstream>

#include "stir/TextWriter.h"

/* Warning: vsnprintf is only ISO C99. So your compiler might not have it.
   Visual Studio can be accomodated with the following work-around
*/
#ifdef BOOST_MSVC
#define vsnprintf _vsnprintf
#endif

START_NAMESPACE_STIR

void warning(const char *const s, ...)
{
  va_list ap;
  va_start(ap, s);
  const unsigned size=10000;
  char tmp[size];
  const int returned_size= vsnprintf(tmp,size, s, ap);
  std::stringstream ss;
  va_end(ap);
  if (returned_size < 0)
	  ss << "\nWARNING: error formatting warning message" << std::endl;
  else
  {
	  ss << "\nWARNING: " << tmp << std::endl;
      if (static_cast<unsigned>(returned_size)>=size)
		  ss << "\nWARNING: previous warning message truncated as it exceeds "
		  << size << "bytes" << std::endl;
  }
  writeText(ss.str().c_str(), WARNING_CHANNEL);
}

END_NAMESPACE_STIR
