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
    Crystal Clear Collaboration
    Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd

    This software is distributed under the terms 
    of the GNU Lesser General  Public Licence (LGPL)
    See STIR/LICENSE.txt for details
*/


#include "local/stir/listmode/CListModeDataLMF.h"
#include "local/stir/listmode/CListRecordLMF.h"
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
  : listmode_filename(listmode_filename)
{
  //opening and reading file.cch, filling in structure LMF_cch  

  // string::c_str() returns a "const char *", but LMFcchReader takes a "char*"
  // (which is a bad idea, unless you really want to modify the string)
  // at the moment, I gamble that LMFcchReader does not modify the string
  // TODO check (and ideally change argument types of LMFcchReader)
  if(LMFcchReader(const_cast<char *>(listmode_filename.c_str()))) 
    exit(EXIT_FAILURE);
  
  //opening file.ccs
  pfCCS = open_CCS_file2(listmode_filename.c_str());  /* open the LMF binary file */    
  if(pfCCS==NULL) 
    error("Cannot open list mode file %s",listmode_filename.c_str());

  fseek(pfCCS,0L,0);                      /* find the begin of file */
  //allocate and fill in the encoding header structure 
  pEncoH = readHead(pfCCS);

  // TODO set scanner_ptr somehow
  //fill scanner check list
  // scanCheckList = fill_ScannerCheckList(scanCheckList, pEncoH);
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

  double x1, y1, z1, x2, y2, z2;
  if (!findXYZinLMFfile(pfCCS,
		        &x1, &y1, &z1, &x2, &y2, &z2,
			pEncoH))
     return Succeeded::no;

  CListEventDataLMF event_data;
  event_data.pos1().x() = static_cast<float>(x1);
  event_data.pos1().y() = static_cast<float>(y1);
  event_data.pos1().z() = static_cast<float>(z1);
  event_data.pos2().x() = static_cast<float>(x2);
  event_data.pos2().y() = static_cast<float>(y2);
  event_data.pos2().z() = static_cast<float>(z2);

  // somehow we have to force record to be a coincidence event.
  // this can be done by assigning, but it is rather wasteful. Better make an appropriate member of CListRecordLMF.
  static_cast<CListRecordLMF&>(record) = event_data;
  return Succeeded::yes;
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

// TODO do ftell and fseek really tell/change about the current listmode event
// or is there another LMF function?
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
