//
// $Id$
//
/*!
  \file
  \ingroup ClearPET_utilities
  \brief Implementation of class CListModeDataLMF
  \author Monica Vieira Martins
  \author Christian Morel
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    // TODO adjust copyright
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/listmode/CListModeDataLMF.h"
#include "local/stir/listmode/CListRecordLMF.h"
#include "/opt/lmf/includes/lmf.h" // TODO adjust location
#include "local/stir/ClearPET/LMF_ClearPET.h" // TODO don't know which is needed
#include "local/stir/ClearPET/LMF_Interfile.h" 
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <stdio.h>

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::streamsize;
using std::streampos;
#endif

START_NAMESPACE_STIR

CListModeDataLMF::
CListModeDataLMF(const string& listmode_filename)
  : listmode_filename(listmode_filename),
{
  //opening and reading file.cch, filling in structure LMF_cch  
  if(LMFcchReader(input_file_name)) exit(EXIT_FAILURE);
  
  //opening file.ccs
  pfCCS = open_CCS_file2(listmode_filename.c_str());  /* open the LMF binary file */    
  fseek(pfCCS,0L,0);                      /* find the begin of file */
  //allocate and fill in the encoding header structure 
  pEncoH = readHead(pfCCS);

  //fill scanner check list
  scanCheckList = fill_ScannerCheckList(scanCheckList, pEncoH);
  // TODO set scanner_ptr somehow
}

CListModeDataLMF::
~CListModeDataLMF()
{
  if(pfCCS) {
    fclose(pfCCS);
  }

  LMFcchReaderDestructor();
  destroyReadHead();//ccsReadHeadDestructor
  destroy_findXYZinLMFfile(pEncoH);
}

shared_ptr <CListRecord> 
CListModeDataLMF::
get_empty_record_sptr() const
{
  return new CListRecordLMF;
}

Succeeded
CListModeDataLMF::
get_next_record(CListRecord& record) const
{
  if (is_null_ptr(pfCCS))
    return Succeeded::no;
  // check type
  assert(dynamic_cast<CListRecordLMF *>(&record) != 0);
  // TODO ignores time    

  // somehow we have to force record to be a coincidence event.
  // this can be done by assigning, but it is rather wasteful. Better make an appropriate member of CListRecordLMF.
  static_cast<CListRecordLMF&>(record) = CListEventDataLMF();
  return
     findXYZinLMFfile(pfCCS,
		      &record.event().pos1().x(),&record.event().pos1().y(),&record.event().pos1().z(),
		      &record.event().pos2().x(),&record.event().pos2().y(),&record.event().pos2().z(),
		      pEncoH) ?
     Succeeded::yes : Succeeded::no;

}

Succeeded
CListModeDataLMF::
reset()
{
  if (is_null_ptr(pfCCS))
    return Succeeded::no;

  if (!fseek(pfCCS,0L,0))                      /* find the begin of file */
    return Succeeded::no;
  else
    return Succeeded::yes;
}


CListModeData::SavedPosition
CListModeDataLMF::
save_get_position() 
{
  assert(!is_null_ptr(pfCCS));
  // TODO should somehow check if ftell() worked and return an error if it didn't
  const unsigned long pos = ftell(pfCCS);
  saved_get_positions.push_back(pos);
  return saved_get_positions.size()-1;
} 

Succeeded
CListModeDataLMF::
set_get_position(const CListModeDataLMF::SavedPosition& pos)
{
  if (is_null_ptr(pfCCS))
    return Succeeded::no;

  assert(pos < saved_get_positions.size());
  if (fseek(pfCCS, saved_get_positions[pos], 0))
    return Succeeded::no;
  else
    return Succeeded::yes;
}
#if 0
vector<unsigned long> 
CListModeDataLMF::
get_saved_get_positions() const
{
  return saved_get_positions;
}

void 
CListModeDataLMF::
set_saved_get_positions(const vector<unsigned long>& poss)
{
  saved_get_positions = poss;
}
#endif
END_NAMESPACE_STIR
