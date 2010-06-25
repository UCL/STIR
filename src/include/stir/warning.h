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

START_NAMESPACE_STIR

//! Print warning with format string a la \c printf
/*! \ingroup buildblock

  The arguments are the same as if you would call printf(). The warning message is written to stderr,
  preceeded by "WARNING:".

  \todo As opposed to using printf-style calling sequence, use a syntax like stir::info

  \par Example
  \code
  warning("Cannot open file %s, but I will work around it", filename);
  \endcode
*/
void
warning(const char *const s, ...);

END_NAMESPACE_STIR
#endif
