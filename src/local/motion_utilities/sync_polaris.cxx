//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This file is for internal GE use only
*/
/*!
  \file
  \ingroup listmode
  \brief Utility to synchronise Polaris data with a list mode file

  \author Kris Thielemans
  $Date$
  $Revision$

  \see RigidObject3DMotionFromPolaris 
  \warning This will change dramatically when using new Polaris acquisition software.
  \par Usage:
  \verbatim
  sync_polaris somefile.mt listmode_filename_prefix
  \endverbatim
  where the list mode data is specified as for \c lm_to_projdata (i.e. without 
  <tt>_1.lm</tt> for ECAT list mode data.
}
*/

#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
#include "stir/listmode/CListModeData.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include <iostream>

USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  const double initial_max_time_offset_deviation = -10E37;
  double max_time_offset_deviation = initial_max_time_offset_deviation;
  if (argc>2 &&  strcmp(argv[1], "--max_time_offset_deviation")==0)
    {
      max_time_offset_deviation = atof(argv[2]);
      argc-=2; argv+=2;
    }
      
  cerr << argc;
  if (argc!=3) {
    std::cerr << "Usage:\n" << argv[0] << "\\\n"
	      << "\t[--max_time_offset_deviation value ] \\\n"
	      << "\tpolarisfile.mt listmode_filename_prefix\n"
	 << "\twhere the list mode data is specified as for lm_to_projdata\n"
	 << "\t(i.e. without _1.lm for ECAT list mode data.\n";
    return EXIT_FAILURE;
  }
  const char * const polaris_filename = argv[1];
  const char * const list_mode_filename = argv[2];

  RigidObject3DMotionFromPolaris polaris_motion;

  if (polaris_motion.set_mt_file(polaris_filename) == Succeeded::no)
    return EXIT_FAILURE;

  shared_ptr<CListModeData> lm_data_ptr =
    CListModeData::read_from_file(list_mode_filename);

  if (max_time_offset_deviation!=initial_max_time_offset_deviation)
    polaris_motion.set_max_time_offset_deviation(max_time_offset_deviation);

  if (polaris_motion.synchronise(*lm_data_ptr) == Succeeded::no)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
