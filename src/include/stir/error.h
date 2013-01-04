//
// $Id$
//
#ifndef __stir_error_H__
#define __stir_error_H__
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2010-06-25, Hammersmith Imanet Ltd
    Copyright (C) 2013-01-01 - $Date$, Kris Thielemans
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
  \brief Declaration of stir::error()

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/common.h"
#include <iostream>
#include <sstream>

START_NAMESPACE_STIR

//! Print error with format string a la \c printf and throw exception
/*! \ingroup buildblock

  The arguments are the same as if you would call printf(). The error message is written to stderr,
  preceeded by "ERROR:", a std::string is constructed with the error message, and 
  <code>throw</code> is called with the string as argument.

  Note that because we throw an exception, the caller can catch it. Prior to STIR 2.1, this was
  not possible as stir::error directly called std::exit.

  \todo As opposed to using printf-style calling sequence, use a syntax like stir::info

  \par Example
  \code
  error("Error opening file %s", filename);
  \endcode

  \deprecated (use 1 argument version instead)
*/
void
error(const char *const s, ...);


//! Use this function for writing error messages and throwing an exception
/*! \ingroup buildblock

  The argument is expected to be a string, but could be anything for which
  std::ostream::operator\<\< would work.

  This function currently first writes a newline, then \c ERROR:, then \c string
  and then another newline to std::cerr. Then it throws an exception (of type string).

  \todo At a later stage, it will also write to a log-file.

  \c boost::format is useful in this context.

  \par Example
  \code
  error(boost::format("Incorrect number of subsets: %d") % num_subsets);

  error("This does not work");
  \endcode
*/

template <class STRING>
inline void
error(const STRING& string)
{
  std::stringstream sstr;
  sstr << "\nERROR: "
	    << string
	    << std::endl;
  std::cerr << sstr.str();
  throw sstr.str();
}

END_NAMESPACE_STIR
#endif
