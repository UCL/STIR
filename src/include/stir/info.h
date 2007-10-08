//
// $Id$
//
#ifndef __stir_info_H__
#define __stir_info_H__
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
  \brief Declaration of stir::info()

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/common.h"
#include <iostream>

START_NAMESPACE_STIR

//! Use this function for writing informational messages
/*! \ingroup buildblock

  The argument is expected to be a string, but could be anything for which
  std::ostream::operator\<\< would work.

  This function currently first writes a newline, then \c INFO, then \c string
  and then another newline to std::cerr. 

  \todo At a later stage, it will also write to a log-file.

  \c boost::format is useful in this context.

  \par Example
  \code
  info(boost::format("Running sub-iteration %" % subiter_num));

  info("Running a really complicated algorithm"));
  \endcode
*/
template <class STRING>
void
info(const STRING& string)
{
  std::cerr << "\nINFO: "
	    << string
	    << std::endl;
}

END_NAMESPACE_STIR
#endif
