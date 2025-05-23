//
//
/*!
  \file
  \ingroup listmode
  \brief Implementations of class stir::CListEvent.
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "stir/listmode/CListRecord.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

Succeeded 
CListEvent::
set_prompt(const bool)
{
  return Succeeded::no; 
}

END_NAMESPACE_STIR
