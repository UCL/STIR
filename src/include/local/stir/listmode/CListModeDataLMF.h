//
//
/*!
  \file
  \ingroup ClearPET_utilities
  \brief Declaration of class stir::CListModeDataLMF
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2004, Hammersmith Imanet Ltd

    This software is distributed under the terms 
    of the GNU Lesser General  Public Licence (LGPL)
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListModeDataLMF_H__
#define __stir_listmode_CListModeDataLMF_H__

#include "stir/listmode/CListModeData.h"
#include "stir/shared_ptr.h"
#include "LMF/lmf.h" // TODO adjust location
//#include "LMF/LMF_ClearPET.h" // TODO don't know which is needed
//#include "LMF/LMF_Interfile.h" 

#include <stdio.h>
#include <string>
#include <vector>

START_NAMESPACE_STIR


//! A class that reads the listmode data from an LMF file
class CListModeDataLMF : public CListModeData
{
public:

  //! Constructor taking a filename
  CListModeDataLMF(const std::string& listmode_filename);

  // Destructor closes the file and destroys various structures
  ~CListModeDataLMF();

  virtual std::time_t 
    get_scan_start_time_in_secs_since_1970() const
  { return std::time_t(-1); } // TODO

  virtual 
    shared_ptr <CListRecord> get_empty_record_sptr() const;


  //! LMF listmode data stores delayed events as well (as opposed to prompts)
  virtual bool has_delayeds() const 
    { return true; } // TODO always?

  virtual 
    Succeeded get_next_record(CListRecord& event) const;

  virtual 
    Succeeded reset();

  virtual 
    SavedPosition save_get_position();

  virtual 
    Succeeded set_get_position(const SavedPosition&);

private:

  string listmode_filename;
  // TODO we really want to make this a shared_ptr I think to avoid memory leaks when throwing exceptions
  struct LMF_ccs_encodingHeader *pEncoH;
  FILE *pfCCS;                                

  // possibly use this from LMF2Projection
  // SCANNER_CHECK_LIST scanCheckList;
  std::vector<unsigned long> saved_get_positions;

};

END_NAMESPACE_STIR

#endif
