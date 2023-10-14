//
//
/*!
  \file
  \ingroup projdata

  \brief Definition of class stir::ViewSegmentNumbers, alias to stir::ViewgramIndices

  \author Kris Thielemans

*/
/*
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ViewSegmentNumbers_h__
#define __stir_ViewSegmentNumbers_h__

#include "stir/ViewgramIndices.h"

START_NAMESPACE_STIR
//! alias for ViewgramIndices
/*!
  For backwards compatibility only.

  \deprecated
*/
// Note: needs to be a class due to forward declarations
class ViewSegmentNumbers : public ViewgramIndices
{
  using ViewgramIndices::ViewgramIndices;
};

END_NAMESPACE_STIR

#endif
