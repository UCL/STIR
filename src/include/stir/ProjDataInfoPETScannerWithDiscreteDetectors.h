/*
    Copyright (C) 2026, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata

  \brief Declaration of class stir::ProjDataInfoPETScannerWithDiscreteDetectors

  \author Kris Thielemans

*/
#ifndef __stir_ProjDataInfoPETScannerWithDiscreteDetectors_H__
#define __stir_ProjDataInfoPETScannerWithDiscreteDetectors_H__

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

START_NAMESPACE_STIR

/*!
  \ingroup projdata
  \brief PET Projection data info for data that correspond directly to discrete detectors (which are not arc-corrected).

  The ProjDataInfo hierarchy is currently badly designed with ProjDataInfoGenericNoArcCorr
  derived from ProjDataInfoCylindricalNoArcCorr. See ProjDataInfoGenericNoArcCorr and
  https://github.com/UCL/STIR/issues/1307.

  Using the current type allows future-proofing.
  */
using ProjDataInfoPETScannerWithDiscreteDetectors = ProjDataInfoCylindricalNoArcCorr;

END_NAMESPACE_STIR

#endif
