//
// $Id$
//
/*!
  \file 
  \ingroup motion

  \brief Implementation of class Polaris_MT_File
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/Succeeded.h"
#include <fstream>
#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::ifstream;
#endif


START_NAMESPACE_STIR  

Polaris_MT_File::Polaris_MT_File(const std::string& mt_filename)
{
  ifstream mt_stream(mt_filename.c_str());
  if (!mt_stream)
  {
    error( "Polaris_MT_File: error opening file %s - Does it Exist?", mt_filename.c_str()) ;
  }
  
  const unsigned int MAX_STRING_LENGTH=512;
  char DataStr[MAX_STRING_LENGTH];
  /* Read opening line */
  {
    // it's of the following format
    //23/5/2003 18:18:32 - 11 1 966_IRSL_II
    if ( !mt_stream.getline( DataStr, MAX_STRING_LENGTH) )
      {
	error("Polaris_MT_File: error reading Line 1 of file %s", 
	      mt_filename.c_str());
      }
#if 0
    int v1, v2;
    char toolkit[MAX_STRING_LENGTH];
    std::tm start_time_tm;
    if (sscanf(DataStr, "%d/%d/%d %d:%d:%d - %d %d %s",
	   &start_time_tm.tm_mday,
	   &start_time_tm.tm_mon,
	   &start_time_tm.tm_year,
	   &start_time_tm.tm_hour,
	   &start_time_tm.tm_min,
	   &start_time_tm.tm_sec,
	   &v1,
	   &v2,
	   toolkit) != 9)
      error("Polaris_MT_File: error parsing first line of file %s",
	    mt_filename.c_str());
  
    start_time_in_secs_since_1970 = 
	mktime(&start_time_tm);

    std::cout << "\nPolaris .mt file info:"
	      << "\n\tDate: " << asctime(&start_time_tm)
	      << "\n\tToolkit: " << toolkit
	      << "\n\tVersion info (?) " << v1 << ' ' << v2 << std::endl;
#endif
  }
  Record record; 
  while (!mt_stream.eof() &&
	 mt_stream.getline( DataStr, MAX_STRING_LENGTH))
    {
      /* Extract elements from string */
      if (sscanf( DataStr, "%lf %u %c %f %f %f %f %f %f %f %f", 
		  &record.sample_time, &record.rand_num, &record.total_num, 
		  &record.quat[1], &record.quat[2], &record.quat[3], &record.quat[4], 
		  &record.trans.x(), &record.trans.y(), &record.trans.z(), 
		  &record.rms ) ==11)
	{
	  // normalise the quaternion. It is only approximately
	  // normalised in the mt file (probably just due to
	  // truncation of the floats etc.)
	  record.quat.normalise();
	  vector_of_records.push_back(record);
	  vector_of_tags.push_back(record);
	}
      else if (sscanf( DataStr, "%lf %u %c : ---- Missing ----",
		       &record.sample_time, &record.rand_num, &record.total_num
		       ) ==3)
	{
	  vector_of_tags.push_back(record);
	}
      else
	{
	  warning("\nPolaris_MT_File: skipping (as I cannot decode) the following line:\n"
		  "'%s'\n",
		  DataStr);
	}
    }

  mt_stream.close();
} 

std::time_t
Polaris_MT_File::
get_start_time_in_secs_since_1970()
{
  return  start_time_in_secs_since_1970;
}

Polaris_MT_File::Record 
Polaris_MT_File::operator[](unsigned int in) const
{
  assert(in>=0);
  assert(in<vector_of_records.size());
  return vector_of_records[in];
}


END_NAMESPACE_STIR  

