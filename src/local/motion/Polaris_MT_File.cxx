//
// $Id$
//
/*!
  \file 
  \ingroup motion

  \brief Implementation of class stir::Polaris_MT_File
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    GE Internal use only
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
  if ( !mt_stream.getline( DataStr, MAX_STRING_LENGTH) )
    {
      error("Polaris_MT_File: error reading Line 1 of file %s", 
	    mt_filename.c_str());
    }

  // find out which file format this file was written in

  if (strncmp(DataStr, "Collection Data", 14)==0)
    {
      // Output of NDI Toolviewer
      // format of first line
      // Collection Data : Port 0B  ( ICL          S/N: 3608C803 Type: 01000000 Rev: 000 )
      read_NDI_Toolviewer_mt_file(mt_filename, mt_stream, DataStr);
    }
  else
    {
      // it's of the following format
      //23/5/2003 18:18:32 - 11 1 966_IRSL_II
      read_Peter_Bloomfield_mt_file(mt_filename, mt_stream, DataStr);
    }
  mt_stream.close();
} 

void
Polaris_MT_File::
read_Peter_Bloomfield_mt_file(const std::string& mt_filename, std::istream& mt_stream, const char * const first_line)
{
  const unsigned int MAX_STRING_LENGTH=512;
  // parse first line
  {
    int v1, v2;
    char toolkit[MAX_STRING_LENGTH];
    std::tm start_time_tm;
    if (sscanf(first_line, "%d/%d/%d %d:%d:%d - %d %d %s",
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

    start_time_tm.tm_mon -= 1;
    start_time_tm.tm_year -= 1900;
    start_time_tm.tm_isdst = -1;

    start_time_in_secs_since_1970 = 
	mktime(&start_time_tm);

    if (start_time_in_secs_since_1970 == std::time_t(-1))
      error("Polaris_MT_File: error interpreting data/time in first line of mt file %s",
	    mt_filename.c_str());

    std::cout << "\nPolaris .mt file info:"
	      << "\n\tDate: " << asctime(&start_time_tm) 
	      << "\t\twhich is " << start_time_in_secs_since_1970 << " secs since 1970 UTC" 
	      << "\n\tToolkit: " << toolkit
	      << "\n\tVersion info (?): " << v1 << ' ' << v2 << std::endl;

  }
  Record record; 
  char DataStr[MAX_STRING_LENGTH];
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
	  // Peter's code only recorded inside FOV events
	  record.out_of_FOV = 0;
	  // Petere's code did not record the frame
	  record.frame_num = 0; // TODO might want to use the 60Hz convention to convert sample_time to frame

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


}



void
Polaris_MT_File::
read_NDI_Toolviewer_mt_file(const std::string& mt_filename, std::istream& mt_stream, const char * const first_line)
{
  const unsigned int MAX_STRING_LENGTH=512;
  char DataStr[MAX_STRING_LENGTH];

  // parse second line with the data
  // format:
  // [June 25, 2010  04:00PM]
  mt_stream.getline( DataStr, MAX_STRING_LENGTH);
  {
    std::tm start_time_tm;
    start_time_tm.tm_sec = 0;
    if (strptime(DataStr, "[%b %d,%Y%I:%M%p]",
		 &start_time_tm) == NULL)
      error("Polaris_MT_File: error parsing date line of file %s",
	    mt_filename.c_str());

    //start_time_tm.tm_mon -= 1;
    //start_time_tm.tm_year -= 1900;
    start_time_tm.tm_isdst = -1;

    start_time_in_secs_since_1970 = 
	mktime(&start_time_tm);

    if (start_time_in_secs_since_1970 == std::time_t(-1))
      error("Polaris_MT_File: error interpreting data/time in 3rd line of mt file %s",
	    mt_filename.c_str());

    std::cout << "\nPolaris .mt file info:"
	      << "\n\tDate: " << asctime(&start_time_tm) 
	      << "\t\twhich is " << start_time_in_secs_since_1970 << " secs since 1970 UTC"  << std::endl;

  }
  // skip empty line
  mt_stream.getline( DataStr, MAX_STRING_LENGTH);

  // skip line containing "frame, Q0, ...
  mt_stream.getline( DataStr, MAX_STRING_LENGTH);


  Record record; 
  while (!mt_stream.eof() &&
	 mt_stream.getline( DataStr, MAX_STRING_LENGTH))
    {
      /* Extract elements from string */
      // Frame,         Q0,         Qx,         Qy,         Qz,          x,          y,          z,      Error,        OOV
      if (sscanf( DataStr, "%u,%f,%f,%f,%f,%f,%f,%f,%f,%u", 
		  &record.frame_num,
		  &record.quat[1], &record.quat[2], &record.quat[3], &record.quat[4], 
		  &record.trans.x(), &record.trans.y(), &record.trans.z(), 
		  &record.rms,
		  &record.out_of_FOV ) ==10)
	{
	  // convert Polaris frame to time
	  // frame runs at a 60Hz clock
	  // set time of first sample to 0
	  if (vector_of_records.size()==0)
	    record.sample_time = 0;
	  else
	    record.sample_time = double(record.frame_num - vector_of_records[0].frame_num)/60;

	  // Toolviewer does not store random numbers sent to listmode
	  record.rand_num = 0;
	  // normalise the quaternion. It is only approximately
	  // normalised in the mt file (probably just due to
	  // truncation of the floats etc.)
	  record.quat.normalise();
	  vector_of_records.push_back(record);
	  // vector_of_tags.push_back(record);
	}
      else if (sscanf( DataStr, "%u,    MISSING",
		       &record.frame_num
		       ) ==1)
	{
	  // just ignore these
	}
      else
	{
	  warning("\nPolaris_MT_File: skipping (as I cannot decode) the following line:\n"
		  "'%s'\n",
		  DataStr);
	}
    }

  warning("NDI Toolviewer output file does not seem to specify the seconds of the acquisition time");
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

