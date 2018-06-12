//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListRecordWithGatingInput which
  is used for list mode data.
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_listmode_CListRecordWithGatingInput_H__
#define __stir_listmode_CListRecordWithGatingInput_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListGatingInput.h"

START_NAMESPACE_STIR

class CListRecordWithGatingInput : public CListRecord
{
 public:
  virtual bool is_gating_input() const { return false; }
  virtual CListGatingInput&  gating_input() = 0; 
  virtual const CListGatingInput&  gating_input() const = 0; 
};

END_NAMESPACE_STIR

#endif
