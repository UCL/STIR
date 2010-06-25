//
// $Id$
//
#ifndef __stir_error_H__
#define __stir_error_H__
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
*/
void
error(const char *const s, ...);

END_NAMESPACE_STIR
#endif
