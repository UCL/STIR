/*
    Copyright (C) 2020, UCL
    Copyright (C) 2020, UKRI    
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
#ifndef __stir_deprecated_H__
#define __stir_deprecated_H__
/*!
  \file 
  \ingroup buildblock
  \brief This file declares a deprecation macro.
*/
START_NAMESPACE_STIR

//! Deprecation macro
#if defined(__GNUC__) || defined(__clang__)
#define STIR_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define STIR_DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define STIR_DEPRECATED
#endif

END_NAMESPACE_STIR


#endif // __stir_deprecated_H__