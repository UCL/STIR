//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Declaration of class CListModeData
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListModeData_H__
#define __stir_listmode_CListModeData_H__

#include "stir/common.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR
class CListRecord;
class Succeeded;

class CListModeData
{
public:

  //! Attempts to get a CListModeData object from a file
  static CListModeData* read_from_file(const string& filename);

  virtual
    ~CListModeData();

  //! Gets the next record in the listmode sequence
  virtual 
    Succeeded get_next_record(CListRecord& event) const = 0;

  //! Call this function if you want to re-start reading at the beginning.
  virtual 
    Succeeded reset() = 0;
};

END_NAMESPACE_STIR

#endif
