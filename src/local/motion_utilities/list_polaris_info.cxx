//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This file is for internal GE use only
*/
/*!
  \file
  \ingroup motion
  \brief Utility to list information from a Polaris file

  \author Kris Thielemans
  $Date$
  $Revision$
  \par Usage:
  \verbatim
  list_polaris_info somefile.mt
  \endverbatim
*/

#include "local/stir/motion/Polaris_MT_File.h"
#include <iostream>

USING_NAMESPACE_STIR

static void print_usage_and_exit(const char * const prog_name)
{
  std::cerr << "Usage:\n" << prog_name << "\\\n"
	    << "\t[--mask-for-tags value ] \\\n"
	    << "\tpolarisfile.mt \n";
  exit(EXIT_FAILURE);
}

int main(int argc, char * argv[])
{
  const char * const prog_name = argv[0];

  unsigned int mask_for_tags = 0xfffffff;
  while (argc>2 && argv[1][0] == '-')
    {
      if (strcmp(argv[1], "--mask-for-tags")==0)
	{
	  mask_for_tags = atoi(argv[2]);
	  argc-=2; argv+=2;
	}
      else
	{
	  print_usage_and_exit(prog_name);
	}
    }

      
  if (argc!=2) {
    print_usage_and_exit(prog_name);
  }
  const char * const polaris_filename = argv[1];

  Polaris_MT_File polaris_data(polaris_filename);

  std::time_t start_time_secs = polaris_data.get_start_time_in_secs_since_1970();
  char * start_time_str = ctime(&start_time_secs); // use internal pointer provided by ctime (string is ended by newline)

  std::cout << "\nInformation for " << polaris_filename
	    << "\nPolaris tracking start at " << start_time_secs << " secs since 1970 UTC, "
	    << "\n   which is " << start_time_str 
	    << "   in your local time zone"
	    << "\nNumber of samples: " << polaris_data.num_samples()
	    << "\nNumber of tags sent to scanner (recorded in the tracking file): " << polaris_data.num_tags();

  if (polaris_data.num_samples()>1)
    {
      std::cout << "\nFirst sample (in \"polaris\" secs) at: " << polaris_data.begin()->sample_time
		<< "\nLast sample  (in \"polaris\" secs) at : " << (polaris_data.end()-1)->sample_time
		<< "\nInterval between first and last sample (in secs) : " << (polaris_data.end()-1)->sample_time - polaris_data.begin()->sample_time;
    }

  std::cout << std::endl;


  return EXIT_SUCCESS;
}
