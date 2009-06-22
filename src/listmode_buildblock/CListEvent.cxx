//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Implementations of class stir::CListEvent.
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
