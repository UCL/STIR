/*
    Copyright (C) 2020, UCL
    Copyright (C) 2020, UKRI
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_deprecated_H__
#define __stir_deprecated_H__
/*!
  \file
  \ingroup buildblock
  \brief This file declares a deprecation macro.
*/
#include "stir/common.h"

START_NAMESPACE_STIR

//! Deprecation macro
#if defined(__GNUC__) || defined(__clang__)
#  define STIR_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#  define STIR_DEPRECATED __declspec(deprecated)
#else
#  pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#  define STIR_DEPRECATED
#endif

END_NAMESPACE_STIR

#endif // __stir_deprecated_H__
