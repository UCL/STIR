//
//
/*!
  \file 
  \ingroup listmode

  \brief Program to bin listmode data to projection data using random rejection of counts (uses stir::LmToProjDataWithRandomRejection)
 
  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2003- 2012, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/listmode/LmToProjDataWithRandomRejection.h"

USING_NAMESPACE_STIR



int main(int argc, char * argv[])
{
  
  if ((argc<1) || (argc>3)) {
    std::cerr << "Usage: " << argv[0] << " [par_file fraction_of_counts_to_keep]]\n";
    exit(EXIT_FAILURE);
  }

  LmToProjDataWithRandomRejection<LmToProjData> 
    application(argc>=2 ? argv[1] : 0);
  if (argc==3)
    application.set_reject_if_above(float(atof(argv[2])));
  application.process_data();
  return EXIT_SUCCESS;
}
