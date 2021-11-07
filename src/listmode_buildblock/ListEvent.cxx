//
//
/*!
  \file
  \ingroup listmode
  \brief Implementations of class stir::ListEvent.
    
  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2019, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "stir/listmode/ListRecord.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

void 
ListEvent::
get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  bin = proj_data_info.get_bin(get_LOR());
}

END_NAMESPACE_STIR

