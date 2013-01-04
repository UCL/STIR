//
// $Id$
//
#ifndef __stir_warning_H__
#define __stir_warning_H__
/*
    Copyright (C) 2010- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.
    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief Declaration of stir::warning()

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/common.h"
#include <iostream>

START_NAMESPACE_STIR

//! Print warning with format string a la \c printf
/*! \ingroup buildblock

  The arguments are the same as if you would call printf(). The warning message is written to stderr,
  preceeded by "WARNING:".

  \par Example
  \code
  warning("Cannot open file %s, but I will work around it", filename);
  \endcode

  \deprecated (use 1 argument version instead)
*/
void
warning(const char *const s, ...);


//! Use this function for writing warning messages
/*! \ingroup buildblock

  The argument is expected to be a string, but could be anything for which
  std::ostream::operator\<\< would work.

  This function currently first writes a newline, then \c WARNING:, then \c string
  and then another newline to std::cerr. 

  \todo At a later stage, it will also write to a log-file.

  \c boost::format is useful in this context.

  \par Example
  \code
  warning(boost::format("Type is like this: %1%. Not sure if that will work.") % projdata_info.parameter_info());

  warning("This might not work");
  \endcode
*/

template <class STRING>
inline void
warning(const STRING& string)
{
  std::cerr << "\nWARNING: "
	    << string
	    << std::endl;
}
END_NAMESPACE_STIR
#endif
