/*
 *  Copyright (C) 2019 University of Hull
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
/*!
  \file
  \ingroup listmode SimSET
  \brief Implementation of class stir::InputStreamFromSimSET

  \author Nikos Efthimiou
*/

#include "stir/listmode/CListRecord.h"

START_NAMESPACE_STIR

bool CListRecordSimSET::is_time() const
{ return true; }

bool CListRecordSimSET::is_event() const
{ return true; }

bool CListRecordSimSET::is_full_event() const
{ return true; }

END_NAMESPACE_STIR
