//
//
#ifndef __stir_info_H__
#define __stir_info_H__
/*
    Copyright (C) 2006- 2013, Hammersmith Imanet Ltd
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief Declaration of stir::info()

  \author Kris Thielemans

*/
#include "stir/common.h"
#include "stir/Verbosity.h"
#include <iostream>
#include <sstream>

#include "TextWriter.h"

START_NAMESPACE_STIR

//! Use this function for writing informational messages
/*! \ingroup buildblock

  The argument is expected to be a string, but could be anything for which
  std::ostream::operator\<\< would work.

  This function currently first writes a newline, then \c INFO, then \c string
  and then another newline to std::cerr.

  \todo At a later stage, it will also write to a log-file.

  \c stir::format is useful in this context.

  \par Example
  \code
  info(format("Running sub-iteration {} of total {}", subiter_num, total));

  info("Running a really complicated algorithm");
  \endcode
*/
template <class STRING>
void
info(const STRING& string, const int verbosity_level = 1)
{
  if (Verbosity::get() >= verbosity_level)
    {
      std::stringstream ss;
      ss << "\nINFO: " << string << std::endl;
      writeText(ss.str().c_str(), INFORMATION_CHANNEL);
    }
}

END_NAMESPACE_STIR
#endif
