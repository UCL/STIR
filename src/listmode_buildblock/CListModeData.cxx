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

#include "stir/listmode/CListModeDataFromStream.h"
#include "stir/listmode/CListModeDataECAT.h"

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
  //return new CListModeDataFromStream(filename);
  return new CListModeDataECAT(filename);
}

const Scanner* 
CListModeData::
get_scanner_ptr() const
{
  return scanner_ptr.get();
}
END_NAMESPACE_STIR
