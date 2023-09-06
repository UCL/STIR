/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, 2021, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

*/
/*!
  \file 
  \ingroup listmode_utilities

  \brief Program to show info about listmode data
 
  \author Kris Thielemans
  \author Daniel Deidda
*/


#include "stir/listmode/ListRecord.h"
#include "stir/listmode/ListEvent.h"
#include "stir/listmode/ListTime.h"
#include "stir/listmode/ListGatingInput.h"
#include "stir/listmode/ListRecordWithGatingInput.h"
#include "stir/listmode/ListModeData.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/Succeeded.h"
#include "stir/IO/read_from_file.h"

#include "stir/Scanner.h"
#include <iostream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
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
  bool list_event_LOR=false;
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
      else if (strcmp(argv[0], "--event-LOR")==0 || strcmp(argv[0], "--SPECT-event")==0)
        {
          list_event_LOR = atoi(argv[1])!=0;
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
           << "--coincidence 0|1  0|1): list coincidence event info or not (default: 0)\n"
           << "--event-LOR 0|1 : ((identical to --SPECT-event) list LOR end-points if coincidence/gamma event or not (default: 0)\n"
           << "--unknown 0|1 : list if event of unknown type encountered or not (default: 0)\n"
           << "--num-events-to-list <num> : limit number of events written to stdout\n"
           << "\nNote that for some PET scanners, coincidences are listed with crystal info.\n"
           << "For others, you should list LOR coordinates (as well) as the 'coincidence' option will only list prompt/delayed info.\n";
      return EXIT_FAILURE;
    }

  if ( list_event_LOR)
    cout << "LORs will be listed as 2 points (z1,y1,x1)-(z2,y2,x2).\n";

  shared_ptr<ListModeData> lm_data_ptr(read_from_file<ListModeData>(argv[0]));

  cout << "Scanner: " << lm_data_ptr->get_scanner_ptr()->get_name() << endl;

  unsigned long num_listed_events = 0;
  {      
    // loop over all events in the listmode file
    shared_ptr <ListRecord> record_sptr = lm_data_ptr->get_empty_record_sptr();
    ListRecord& record = *record_sptr;

    while (num_events_to_list==0 || num_events_to_list!=num_listed_events)
      {
        bool recognised = false;
        bool listed = false;
//      std::cout<<"ciao"<<std::endl;
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
          ListRecordWithGatingInput * record_ptr = dynamic_cast<ListRecordWithGatingInput *>(&record);
          if (record_ptr!=0 && record_ptr->is_gating_input())
            {
              recognised=true;
              if (list_gating)
                {
                  cout << "Gating " << std::hex << record_ptr->gating_input().get_gating() << std::dec;
                  listed = true; 
                }
            }
        }
        if (record.is_event())
        {
          recognised=true;
          if (list_coincidence)
            {
              if (auto event_ptr = 
                   dynamic_cast<CListEvent *>(&record.event()))
                {
                  cout << "Coincidence " << (event_ptr->is_prompt() ? "p " : "d ");
                }
              if (auto event_ptr = 
                   dynamic_cast<CListEventCylindricalScannerWithDiscreteDetectors *>(&record.event()))
                {
                  DetectionPositionPair<> det_pos;
                  event_ptr->get_detection_position(det_pos);
                  cout << "(c:" << det_pos.pos1().tangential_coord()
                       << ",r:" << det_pos.pos1().axial_coord()
                       << ",l:" << det_pos.pos1().radial_coord()
                       << ")-"
                       << "(c:" << det_pos.pos2().tangential_coord()
                       << ",r:" << det_pos.pos2().axial_coord()
                       << ",l:" << det_pos.pos2().radial_coord()
                       << ")\t";
                  cout << " TOF-bin: " << det_pos.timing_pos()
                       << " delta time: " << event_ptr->get_delta_time();
                  listed = true; 
                }
            }
          if (list_event_LOR)
            { 
              if (auto event_ptr = 
                   dynamic_cast<CListEvent *>(&record.event())) // cast not necessary, but looks same as above
              if (event_ptr!=0)
                {
                  LORAs2Points<float> lor;
                  lor=event_ptr->get_LOR();
                  cout << " LOR "
                       << "(" << lor.p1().z()
                       << "," << lor.p1().y()
                       << "," << lor.p1().x()
                       << ")-"
                       << "(" << lor.p2().z()
                       << "," << lor.p2().y()
                       << "," << lor.p2().x()
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

