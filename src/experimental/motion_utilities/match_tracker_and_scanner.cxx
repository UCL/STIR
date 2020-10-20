//
//
/*
    Copyright (C) 2003- 2005 , Hammersmith Imanet Ltd
    For GE Internal use only
*/
/*!
  \file
  \ingroup motion_utilities
  \brief A utility  for finding the coordinate transformation between tracker and scanner 
  coordinate systems.

  \par Usage
  \verbatim
  match_tracker_and_scanner par_file
  \endverbatim
  See doxygen documentation of stir::MatchTrackerAndScanner for explanation and
  the format of the parameter file.

  \author Kris Thielemans

  
*/
#include "stir/Succeeded.h"
#include "stir_experimental/motion/MatchTrackerAndScanner.h"

int main(int argc, char ** argv)
{

  if (argc!=1 && argc!=2) {
    cerr << "Usage: " << argv[0] << " \\\n"
	 << "\t[par_file]\n";
    exit(EXIT_FAILURE);
  }
  stir::MatchTrackerAndScanner application(argc==2 ? argv[1] : 0);

  return 
    application.run() == stir::Succeeded::yes ?
    EXIT_SUCCESS : EXIT_FAILURE;
}
