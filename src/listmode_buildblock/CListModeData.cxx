//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Implementation of class stir::CListModeData 
    
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

#include "stir/IO/read_from_file.h"
#include "stir/listmode/CListModeData.h"

START_NAMESPACE_STIR

CListModeData::
CListModeData()
{
}

CListModeData::
~CListModeData()
{}

CListModeData*
CListModeData::
read_from_file(const string& filename)
{
  std::auto_ptr<CListModeData > aptr = 
    stir::read_from_file<CListModeData>(filename);
  return aptr.release();
}

const Scanner* 
CListModeData::
get_scanner_ptr() const
{
  return this->scanner_sptr.get();
}
END_NAMESPACE_STIR
