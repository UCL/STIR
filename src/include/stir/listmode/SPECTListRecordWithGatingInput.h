///
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::SPECTListRecordWithGatingInput which
  is used for list mode data.

  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, University College of London
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

#ifndef __stir_listmode_SPECTListRecordWithGatingInput_H__
#define __stir_listmode_SPECTListRecordWithGatingInput_H__

#include "SPECTListRecord.h"
#include "ListGatingInput.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

class SPECTListRecordWithGatingInput : public virtual SPECTListRecord
{
 public:
  virtual bool is_gating_input() const { return false; }
  virtual ListGatingInput&  gating_input() = 0;
  virtual const ListGatingInput&  gating_input() const = 0;
};

END_NAMESPACE_STIR

#endif
