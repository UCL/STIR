//
//
/*
  Copyright (C) 2005- 2007, Hammersmith Imanet Ltd
  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup utilities
\brief Create 4D projdata from 3D using stir::inverse_SSRB

\author Charalampos Tsoumpas
\author Kris Thielemans


\par Usage:
\code
correct_for_scatter [4D_projdata_filename] [3D_projdata] [4D_template]
Output: 4D Projdata .hs .s files with name proj_data_4D
\endcode

This is a utility program which uses the stir::inverse_SSRB function , in order to create a
4D set of projection data.
*/
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/inverse_SSRB.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include <iostream>
#include <string>
using std::endl;
using std::cout;
using std::cerr;
USING_NAMESPACE_STIR
using namespace std;
/***********************************************************/

int
main(int argc, const char* argv[])
{

  if (argc < 3 || argc > 4)
    {
      cerr << "Usage:" << argv[0] << "\n"
           << "\t[projdata_4D_filename]\n"
           << "\t[projdata_3D]\n"
           << "\t[projdata_4D_template]\n";

      return EXIT_FAILURE;
    }
  shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(argv[3]);
  const ProjDataInfo* proj_data_info_ptr = dynamic_cast<ProjDataInfo const*>(template_proj_data_sptr->get_proj_data_info_sptr());

  const shared_ptr<ProjData> proj_data_3D_sptr = ProjData::read_from_file(argv[2], ios::in);

  if (proj_data_info_ptr == 0 || proj_data_3D_sptr == 0)
    error("Check the input files\n");

  string proj_data_4D_filename(argv[1]);
  ProjDataInterfile proj_data_4D(proj_data_info_ptr->clone(), proj_data_4D_filename, ios::out);

  const Succeeded success = inverse_SSRB(proj_data_4D, *proj_data_3D_sptr);

  return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
