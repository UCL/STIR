//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    For GE internal use only.
*/
/*!
  \file 
  \ingroup listmode_utilities

  \brief Program to show info about listmode data
 
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/


#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListModeData.h"
#include "stir/Succeeded.h"

#include "stir/Scanner.h"
#include <iostream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::hex;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
#endif



USING_NAMESPACE_STIR
#if 0
// TODO this is only good for the Polaris data
void
find_and_store_gate_tag_values_from_lm(vector<float>& lm_time, 
				       vector<unsigned>& lm_gate, 
				       CListModeData& listmode_data,
				       const unsigned long max_num_events)
{
  
  unsigned  LastChannelState=0;
  unsigned  ChState;
  int PulseWidth = 0 ;
  double StartPulseTime=0;
  unsigned long more_events = max_num_events;
  
  // reset listmode to the beginning 
  listmode_data.reset();
  
  shared_ptr <CListRecord> record_sptr = listmode_data.get_empty_record_sptr();
  CListRecord& record = *record_sptr;
  while (max_num_events==0 || more_events!=0)
  {

    if (listmode_data.get_next_record(record) == Succeeded::no) 
    {
       break; //get out of while loop
    }
    if (record.is_time())
    {
      unsigned CurrentChannelState =  record.time().get_gating() ;
      double CurrentTime = record.time().get_time_in_secs();
      
      if ( LastChannelState != CurrentChannelState && CurrentChannelState )
      {
	if ( PulseWidth > 5 ) //TODO get rid of number 5
	{
	  lm_gate.push_back(ChState);
	  lm_time.push_back(StartPulseTime);
	  --more_events;
	}
	LastChannelState = CurrentChannelState ;
	PulseWidth = 0 ;
      }
      else if ( LastChannelState == CurrentChannelState && CurrentChannelState )
      {
	if ( !PulseWidth ) StartPulseTime = CurrentTime ;
	ChState = LastChannelState ;
	PulseWidth += 1 ;
      }
    }
  }
  // reset listmode to the beginning 
  listmode_data.reset();
 
}

#endif

int main(int argc, char *argv[])
{

  if (argc<2 || argc>3)
    {
      cerr << "Usage: " << argv[0] << "lm_filename_prefix [num_time_events_to_list]\n";
      return EXIT_FAILURE;
    }

  shared_ptr<CListModeData> lm_data_ptr =
    CListModeData::read_from_file(argv[1]);
  const unsigned long num_events_to_list =
    argc==3 ? atol(argv[2]) : 0;

  cout << "Scanner: " << lm_data_ptr->get_scanner_ptr()->get_name() << endl;

#if 0
  // only changes in gates (appropriate for Polaris)

  vector<float> lm_time;
  vector<unsigned> lm_gate;
  find_and_store_gate_tag_values_from_lm(lm_time, 
					 lm_gate, 
				         *lm_data_ptr,
					 num_events_to_list);
  vector<float>::const_iterator lm_time_iter = lm_time.begin();
  vector<unsigned>::const_iterator lm_gate_iter = lm_gate.begin();
  for (;lm_time_iter != lm_time.end(); ++lm_time_iter, ++lm_gate_iter)
    {
      cerr << *lm_time_iter << '\t' << hex << *lm_gate_iter << '\n';
    }
  
#else
  // all of them
  unsigned long num_listed_events = 0;
  {      
    // loop over all events in the listmode file
    shared_ptr <CListRecord> record_sptr = lm_data_ptr->get_empty_record_sptr();
    CListRecord& record = *record_sptr;

    while (num_events_to_list==0 || num_events_to_list!=num_listed_events)
      {
	if (lm_data_ptr->get_next_record(record) == Succeeded::no) 
	  {
	    // no more events in file for some reason
	    break; //get out of while loop
	  }
	if (record.is_time())
	  {
	    cout << record.time().get_time_in_secs()
		 << '\t' << hex << record.time().get_gating() << '\n';
	    ++num_listed_events;
	    
	   } 
      }

  }
#endif

  return EXIT_SUCCESS;
}

