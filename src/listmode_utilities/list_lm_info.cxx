/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode_utilities

  \brief A utility that lists info on the listmode data on stdout.

  \par Usage

  <pre>
  list_lm_info [--all  | --exam] listmode_filename
  </pre>
  Add one or more options to print the exam/geometric information.
  If no option is specified, exam info is printed.

  \author Kris Thielemans
*/
#include "stir/listmode/ListModeData.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include <iostream> 
#include <string>

USING_NAMESPACE_STIR

void print_usage_and_exit(const std::string& program_name)
{
  std::cerr<<"Usage: " << program_name << " [--all | --geom | --exam] listmode_file\n"
	   <<"\nAdd one or more options to print the exam/geometric information.\n"
	   <<"\nIf no option is specified, exam info is printed.\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{ 
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  // default values
  bool print_exam = false;
  bool print_geom = false;
  bool no_options = true; // need this for default behaviour

  // first process command line options
  while (argc>0 && argv[0][0]=='-' && argc>=2)
    {
      no_options=false;
      if (strcmp(argv[0], "--all")==0)
	{
	  print_geom = print_exam = true; 
	  --argc; ++argv;
	}
      else if (strcmp(argv[0], "--geom")==0)
	{
	  print_geom = true; 
	  --argc; ++argv;
	}
      else if (strcmp(argv[0], "--exam")==0)
	{
	  print_exam = true; 
	  --argc; ++argv;
	}
      else
	print_usage_and_exit(program_name);
    }
  if (no_options)
    print_exam = true;

  if(argc!=1) 
  {
    print_usage_and_exit(program_name);
  }

  // set filename to last remaining argument
  const std::string filename(argv[0]);

  shared_ptr<ListModeData> lm_data_sptr(read_from_file<ListModeData>(filename));

  if (is_null_ptr(lm_data_sptr))
    {
      warning("Could not read %s", filename.c_str());
      return EXIT_FAILURE;
    }

  if (print_exam)
    std::cout << lm_data_sptr->get_exam_info_sptr()->parameter_info();
  if (print_geom)
    std::cout << lm_data_sptr->get_proj_data_info_sptr()->parameter_info() << std::endl;
  return EXIT_SUCCESS;
}
