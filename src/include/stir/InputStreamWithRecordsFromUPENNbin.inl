/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecords
    
  \author Kris Thielemans
*/
/*
    Copyright (C) 2003-2011, Hammersmith Imanet Ltd
    Copyright (C) 2012-2013, Kris Thielemans
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


#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/shared_ptr.h"
#include "boost/shared_array.hpp"
#include "libgeom.h"
#include "liblist.h"
#include "liboption.h"
#include <fstream>

START_NAMESPACE_STIR
InputStreamWithRecordsFromUPENNbin::
InputStreamWithRecordsFromUPENNbin(std::string _filename,
                                   int keep_type,
                                   int minE, int maxE, 
				   long unsigned int _N)
  : keep_type(keep_type),
    minE(minE),
    maxE(maxE)
{

    filename = _filename;
    if ( !filename.empty() )
    {
        if ( !inputListFile.open( filename.c_str(), std::ios_base::in | std::ios_base::binary ) )
        {
            std::cerr << "error: cannot open file " << filename.c_str() << '\n';
            std::exit( EXIT_FAILURE );
        }

        inputList = &inputListFile;
    }

    if ( !list::decodeHeader( *inputList, &listHeader )
         || !list::isCoincListHeader( listHeader ) )
    {
        std::cerr << "error: cannot read valid header from input list\n";
        std::exit( EXIT_FAILURE );
    }

    N = _N; 
    abrupt_counter = N;
    if (N > 0)
	abrupt = true; 
    eventFormat = list::eventFormat( listHeader );
    eventCodec = new list::EventCodec( eventFormat );
    eventSize = list::eventSize( eventFormat );
    in = new list::InputBuffer( *inputList, eventSize );
    pos = inputListFile.pubseekoff(0, std::ios_base::cur);
}

Succeeded
InputStreamWithRecordsFromUPENNbin::
create_output_file(std::string ofilename)
{
//	std::cout<<"Here0"<<std::endl;
//    list::EventFormat eventFormat = list::Explorer;
    //olistCodec = new list::EventCodec( list::Explorer );
    olistCodec = new list::EventCodec( eventFormat );
  //  std::cout <<"Here1"<< ofilename <<std::endl;
    if ( !ofilename.empty() )
    {
        const std::ios_base::openmode writeonly = std::ios_base::out
                | std::ios_base::binary;
//	std::cout << "Here2"<<std::endl; 
        if ( !outputListFile.open( ofilename.c_str(), writeonly ) )
        {
//		std::cout <<"Here3"<<std::endl;
            std::cerr << "error: cannot create file " << ofilename << '\n';
            std::exit( EXIT_FAILURE );
        }

//	std::cout << "Here4"<<std::endl;
        outputList = &outputListFile;
    }

    if ( !list::encodeHeader( *outputList, listHeader ) )
    {
//	    std::cout<<"Here6"<<std::endl; 
        std::cerr << "error: cannot write header to output list\n";
        std::exit( EXIT_FAILURE );
    }
//	std::cout <<"Here7"<<std::endl;
    out = new list::OutputBuffer(*outputList, eventSize );
    has_output = true;
//	std::cout<<"Here8"<<std::endl; 	
    if(keep_type == 2 && has_output)
    {
        error("You cannot keep delayed events and pass output.");
    }
    return Succeeded::yes;
}

Succeeded
InputStreamWithRecordsFromUPENNbin::
get_next_record(CListRecordPENNbin& record)
{
//    list::Event* event = nullptr;
    int dt;
    int xa ;
    int xb ;
    int za ;
    int zb ;
    int ea;
    int eb;
    bool is_delay = false;
    bool found = false;

#ifdef STIR_OPENMP
#pragma omp critical(LISTMODEIO)
#endif
    while (in->next())
    {
//        event = new list::Event( *eventCodec, in->data() );

        list::Event event( *eventCodec, in->data() );
//        if (is_null_ptr(event))
//            break;

//        dt = event.dt();
//        xa = event.xa();
//        xb = event.xb();
//        za = event.za();
//        zb = event.zb();
//        ea = event.ea();
//        eb = event.eb();

//        if (dt == 11 &&
//                xa == 80 && xb == 357)
//            int nikos = 0;

        // Here we pass the delay events
/*        if(has_output)
        {
            //                    if(!event->isPrompt()) // this will be handled later
            //                    {
            //                        set_record(in->data());
            //                    }
            //                    else {
            current_record = in->data();
//            current_record[0] = *(in->data());
//            current_record[1] = *(in->data() + 1);
//            current_record[2] = *(in->data() + 2);
//            current_record[3] = *(in->data() + 3);
//            current_record[4] = *(in->data() + 4);
//            current_record[5] = *(in->data() + 5);
//            current_record[6] = *(in->data() + 6);
//            current_record[7] = *(in->data() + 7);
            //                    }
        }*/

	if(abrupt)
	{
		abrupt_counter--;
		if (abrupt_counter < 0)
		{
			found = false; 
			break;
		}
	}
	      	
        if(!event.isData() || event.isControl())
        {
            timeout = 0;
//            delete [] event;
//            event = nullptr;
            continue;
        }
        else if(event.ea()<minE || event.ea() > maxE ||
                event.eb()<minE || event.eb() > maxE)
        {
            timeout = 0;
            continue;
        }
        else if((keep_type == 1 || keep_type == 3) && event.isPrompt())
        {
            timeout = 0;
            dt = event.dt();
            xa = event.xa();
            xb = event.xb();
            za = event.za();
            zb = event.zb();
            ea = event.ea();
            eb = event.eb();
            is_delay = false;
            if (xa < 576 && xb < 576 &&
			    //dt >= 0 && 
                 //   za >= 101 && za < 120 && 
		 //   zb >= 101 && zb < 120 &&
                    xa != xb)// && dt >= 0)
            {

                found = true;
                break;
            }else {
                continue;
            }
        }
        else if ((keep_type == 2 || keep_type == 3) && event.isDelay())
        {
            timeout = 0;

            dt = event.dt();
            xa = event.xa();
            xb = event.xb();
            za = event.za();
            zb = event.zb();
            ea = event.ea();
            eb = event.eb();
            is_delay = true;
	    if (xa < 576 && xb < 576 &&
			    //dt >= 0 && 
                   // za >= 101 && za < 120 && 
		   // zb >= 101 && zb < 120 &&
                    xa != xb)// && dt >= 0)
            {
                found = true;
                break;
            }else {
                continue;
            }
        }
        else {
//            delete [] event;
//            event = nullptr;
            continue;
        }
    }

    if(found)
    {
	    if(abrupt)
	    if(abrupt_counter < 0)
		return Succeeded::no;
        return
                record.init_from_data_ptr(is_delay,
                                          dt,
                                          xa, xb,
                                          za, zb,
                                          ea, eb
                                          );
    }
    else {
//        if(event!=nullptr)
//            delete [] event;

        return Succeeded::no;
    }

}

Succeeded
InputStreamWithRecordsFromUPENNbin::
reset()
{
    inputListFile.pubseekpos(pos);
    inputList->pubsync();

    abrupt_counter = N;
    if(!is_null_ptr(in))
    {
        delete in;
        in = new list::InputBuffer( *inputList, eventSize );
    }
    return Succeeded::yes;
}

typename InputStreamWithRecordsFromUPENNbin::SavedPosition
InputStreamWithRecordsFromUPENNbin::
save_get_position() 
{
    int pos = inputListFile.pubseekoff(0, std::ios_base::cur);
    saved_get_positions.push_back(pos);
    return saved_get_positions.size()-1;
} 

Succeeded
InputStreamWithRecordsFromUPENNbin::
set_get_position(const typename InputStreamWithRecordsFromUPENNbin::SavedPosition& pos)
{
//  if (is_null_ptr(stream_ptr))
//    return Succeeded::no;

  assert(pos < saved_get_positions.size());
//  stream_ptr->clear();
//  if (saved_get_positions[pos] == std::streampos(-1))
//    stream_ptr->seekg(0, std::ios::end); // go to eof
//  else
//    stream_ptr->seekg(saved_get_positions[pos]);
  inputListFile.pubseekoff(saved_get_positions[pos], std::ios_base::beg);
    
//  if (!stream_ptr->good())
//    return Succeeded::no;
//  else
    return Succeeded::yes;
}

std::vector<std::streampos> 
InputStreamWithRecordsFromUPENNbin::
get_saved_get_positions() const
{
  return saved_get_positions;
}

void 
InputStreamWithRecordsFromUPENNbin::
set_saved_get_positions(const std::vector<std::streampos>& poss)
{
  saved_get_positions = poss;
}

END_NAMESPACE_STIR
