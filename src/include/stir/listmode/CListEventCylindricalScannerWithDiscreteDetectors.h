//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of class stir::CListEventCylindricalScannerWithDiscreteDetectors

  \author Kris Thielemans

*/
/*
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_listmode_CListEventCylindricalScannerWithDiscreteDetectors_H__
#define __stir_listmode_CListEventCylindricalScannerWithDiscreteDetectors_H__

#include "stir/listmode/CListEventScannerWithDiscreteDetectors.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

START_NAMESPACE_STIR

//! Class for storing and using a coincidence event from a list mode file for a cylindrical scanner
/*! \ingroup listmode
   At present, could just as well be a `typedef`.
*/
class CListEventCylindricalScannerWithDiscreteDetectors
    : public CListEventScannerWithDiscreteDetectors<ProjDataInfoCylindricalNoArcCorr>
{
private:
  typedef CListEventScannerWithDiscreteDetectors<ProjDataInfoCylindricalNoArcCorr> base_type;

public:
  using base_type::base_type;
};

END_NAMESPACE_STIR

//#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.inl"

#endif
