//
// $Id$
//
/*!
  \file
  \ingroup buildblock  
  \brief Implementation of class CListModeData 
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/listmode/CListModeDataFromStream.h"
#include "local/stir/listmode/CListModeDataECAT.h"

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
