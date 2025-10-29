//
//
#ifndef __stir_warning_H__
#define __stir_warning_H__
/*
    Copyright (C) 2010- 2013, Hammersmith Imanet Ltd
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief Declaration of stir::warning()

  \author Kris Thielemans

*/
#include "stir/Verbosity.h"
#include "stir/TextWriter.h"
#include <sstream>

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
void warning(const char* const s, ...);

//! Use this function for writing warning messages
/*! \ingroup buildblock

  The argument is expected to be a string, but could be anything for which
  std::ostream::operator\<\< would work.

  This function currently first writes a newline, then \c WARNING:, then \c string
  and then another newline to std::cerr.

  \todo At a later stage, it will also write to a log-file.

  \c stir::format is useful in this context.

  \par Example
  \code
  warning(format("Type is like this: {}. Not sure if that will work.", projdata_info.parameter_info()));

  warning("This might not work");
  \endcode
*/

template <class STRING>
inline void
warning(const STRING& string, const int verbosity_level = 1)
{
  if (Verbosity::get() >= verbosity_level)
    {
      std::stringstream sstr;
      sstr << "\nWARNING: " << string << std::endl;
      writeText(sstr.str().c_str(), WARNING_CHANNEL);
    }
}
END_NAMESPACE_STIR
#endif
