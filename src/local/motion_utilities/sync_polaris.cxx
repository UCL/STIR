/*!
  \file
  \ingroup listmode
  \brief Utility to synchronise Polaris data with a list mode file

  \author Kris Thielemans
  $Date$
  $Revision$
  
  \par Usage:
  \verbatim
  sync_polaris somefile.mt listmode_filename_prefix
  \endverbatim
  where the list mode data is specified as for \c lm_to_projdata (i.e. without 
  <tt>_1.lm</tt> for ECAT list mode data.
}
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
#include "stir/listmode/CListModeData.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include <iostream>

USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  if (argc!=3) {
    std::cerr << "Usage:\n" << argv[0] << " polarisfile.mt listmode_filename_prefix\n"
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

  if (polaris_motion.synchronise(*lm_data_ptr) == Succeeded::no)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
