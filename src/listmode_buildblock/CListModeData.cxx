//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Implementation of class CListModeData 
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
