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

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
#endif

const unsigned int MAX_STRING_LENGTH=512;

START_NAMESPACE_STIR  

Polaris_MT_File::Polaris_MT_File(const std::string& mt_filename)
{
  ifstream mt_stream(mt_filename.c_str());
  if (!mt_stream)
  {
    error( "\n\t\tError Opening Supplied File %s - Does it Exist?\n", mt_filename.c_str()) ;
  }
  
  /* Read opening line - discard */
  char DataStr[MAX_STRING_LENGTH];
  if ( !mt_stream.getline( DataStr, MAX_STRING_LENGTH) )
  {
    error("\n\t\tError Reading Line 1 of Supplied File %s\n", mt_filename.c_str());
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


Polaris_MT_File::Record 
Polaris_MT_File::operator[](unsigned int in) const
{
  assert(in>=0);
  assert(in<vector_of_records.size());
  return vector_of_records[in];
}


END_NAMESPACE_STIR  

