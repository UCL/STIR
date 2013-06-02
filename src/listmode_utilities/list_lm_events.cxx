//
// $Id$
//
/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
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
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"
#include "stir/IO/read_from_file.h"

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

int main(int argc, char *argv[])
{
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  bool list_time=true;
  bool list_coincidence=false;
  bool list_gating=true;
  bool list_unknown=false;
  unsigned long num_events_to_list = 0;
  while (argc>1 && argv[0][0]=='-')
    {
      if (strcmp(argv[0], "--num-events-to-list")==0)
	{
	  num_events_to_list = atol(argv[1]);
	} 
      else if (strcmp(argv[0], "--time")==0)
        {
          list_time = atoi(argv[1])!=0;
        }
      else if (strcmp(argv[0], "--gating")==0)
        {
          list_gating = atoi(argv[1])!=0;
        }
      else if (strcmp(argv[0], "--coincidence")==0)
        {
          list_coincidence = atoi(argv[1])!=0;
        }
      else if (strcmp(argv[0], "--unknown")==0)
        {
          list_unknown = atoi(argv[1])!=0;
        }
      else
        { 
          cerr << "Unrecognised option\n";
          return EXIT_FAILURE;
        }
      argc-=2; argv+=2;
    }

  if (argc!=1)
    {
      cerr << "Usage: " << program_name << "[options] lm_filename\n"
           << "Options:\n"
           << "--time 0|1 : list time events or not (default: 1)\n"
           << "--gating 0|1 : list gating events or not (default: 1)\n"
           << "--coincidence 0|1 : list coincidence events or not (default: 0)\n"
           << "--unknown 0|1 : list if event of unknown type encountered or not (default: 0)\n"
           << "--num-events-to-list <num> : limit number of events written to stdout\n";
      return EXIT_FAILURE;
    }

  shared_ptr<CListModeData> lm_data_ptr(read_from_file<CListModeData>(argv[0]));  

  cout << "Scanner: " << lm_data_ptr->get_scanner_ptr()->get_name() << endl;

  unsigned long num_listed_events = 0;
  {      
    // loop over all events in the listmode file
    shared_ptr <CListRecord> record_sptr = lm_data_ptr->get_empty_record_sptr();
    CListRecord& record = *record_sptr;

    while (num_events_to_list==0 || num_events_to_list!=num_listed_events)
      {
        bool recognised = false;
        bool listed = false;
	if (lm_data_ptr->get_next_record(record) == Succeeded::no) 
	  {
	    // no more events in file for some reason
	    break; //get out of while loop
	  }
	if (record.is_time())
	  {
            recognised=true;
            if (list_time)
              {
                cout << "Time " << record.time().get_time_in_millisecs();
                listed = true; 
              }
	   } 
        {
          CListRecordWithGatingInput * record_ptr = dynamic_cast<CListRecordWithGatingInput *>(&record);
          if (record_ptr!=0 && record_ptr->is_gating_input())
            {
              recognised=true;
              if (list_gating)
                {
                  cout << "Gating " << hex << record_ptr->gating_input().get_gating();
                  listed = true; 
                }
            }
        }
        if (record.is_event())
        {
          recognised=true;
          if (list_coincidence)
            {CListEventCylindricalScannerWithDiscreteDetectors * event_ptr = 
                dynamic_cast<CListEventCylindricalScannerWithDiscreteDetectors *>(&record.event());
              if (event_ptr!=0)
                {
                  DetectionPositionPair<> det_pos;
                  event_ptr->get_detection_position(det_pos);
                  cout << "Coincidence " << (event_ptr->is_prompt() ? "p " : "d ")
                       << "(c:" << det_pos.pos1().tangential_coord()
                       << ",r:" << det_pos.pos1().axial_coord()
                       << ",l:" << det_pos.pos1().radial_coord()
                       << ")-"
                   << "(c:" << det_pos.pos2().tangential_coord()
                       << ",r:" << det_pos.pos2().axial_coord()
                       << ",l:" << det_pos.pos2().radial_coord()
                       << ")";
                  listed = true; 
                }
            }
        }
        if (!recognised && list_unknown)
        {
          cout << "Unknown type";
          listed = true; 
        }
        if (listed)
          {
            ++num_listed_events;	    
            cout << '\n';
          }
      }
    cout << '\n';

  }

  return EXIT_SUCCESS;
}

