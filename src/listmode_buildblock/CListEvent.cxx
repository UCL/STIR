//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Implementations of class CListEvent.
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This software is distributed under the terms of the GNU Lesser General 
    Public Licence (LGPL).
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

LORAs2Points<float>
CListEvent::
get_LOR() const
{
  LORAs2Points<float> lor;
  get_detection_coordinates(lor.p1(), lor.p2());
  return lor;
}

void 
CListEvent::
get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  bin = proj_data_info.get_bin(get_LOR());
}

END_NAMESPACE_STIR
