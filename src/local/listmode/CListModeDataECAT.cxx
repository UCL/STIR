//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Implementation of class CListModeDataECAT
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/listmode/CListModeDataECAT.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

CListModeDataECAT::
CListModeDataECAT(const string& listmode_filename_prefix)
  : listmode_filename_prefix(listmode_filename_prefix)    
{
  if (open_lm_file(1) == Succeeded::no)
    error("CListModeDataECAT: error opening the first listmode file for filename %s\n",
	  listmode_filename_prefix.c_str());
}

Succeeded
CListModeDataECAT::
open_lm_file(unsigned int new_lm_file) const
{
  if (is_null_ptr(current_lm_data_ptr) || new_lm_file != current_lm_file)
    {
      string filename = listmode_filename_prefix;
      char rest[50];
      sprintf(rest, "_%d.lm", new_lm_file);
      filename += rest;
      cerr << "CListModeDataECAT: opening file " << filename << endl;
      current_lm_data_ptr =
	new CListModeDataFromStream(filename);
      current_lm_file = new_lm_file;
      return Succeeded::yes;
    }
  else
    return current_lm_data_ptr->reset();
}

Succeeded
CListModeDataECAT::
get_next_record(CListRecord& record) const
{
  if (current_lm_data_ptr->get_next_record(record) == Succeeded::yes)
    return Succeeded::yes;
  else
  {
    if (open_lm_file(++current_lm_file) == Succeeded::yes)
      return current_lm_data_ptr->get_next_record(record);
    else
      return Succeeded::no;
  }
}



Succeeded
CListModeDataECAT::
reset()
{
  if (current_lm_file!=1)
    {
      return open_lm_file(1);
    }
  else
    {
      return current_lm_data_ptr->reset();
    }
}

#if 0
unsigned long
CListModeDataECAT::
get_num_records() const
{ 
}

#endif
END_NAMESPACE_STIR
