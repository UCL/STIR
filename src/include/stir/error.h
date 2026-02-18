//
//
#ifndef __stir_error_H__
#define __stir_error_H__
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2010-06-25, Hammersmith Imanet Ltd
    Copyright (C) 2013-01-01 - 2013, Kris Thielemans
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief Declaration of stir::error()

  \author Kris Thielemans

*/
#include "stir/common.h"
#include <iostream>
#include <sstream>

#include "TextWriter.h"

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
void error(const char* const s, ...);

//! Use this function for writing error messages and throwing an exception
/*! \ingroup buildblock

  The argument is expected to be a string, but could be anything for which
  std::ostream::operator\<\< would work.

  This function currently first writes a newline, then \c ERROR:, then \c string
  and then another newline to std::cerr. Then it throws an exception.

  \todo At a later stage, it will also write to a log-file.

  \c stir::format is useful in this context.

  \par Example
  \code
  error(format("Incorrect number of subsets: {}", num_subsets));

  error("This does not work");
  \endcode
*/

template <class STRING>
inline void
error(const STRING& string)
{
  std::stringstream sstr;
  sstr << "\nERROR: " << string << std::endl;
  writeText(sstr.str().c_str(), ERROR_CHANNEL);
  throw std::runtime_error(sstr.str());
}

END_NAMESPACE_STIR
#endif
