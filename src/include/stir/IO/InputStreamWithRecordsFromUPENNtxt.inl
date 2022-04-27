/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecordsFromUPENNtxt
    
  \author Nikos Efthimiou
*/
/*
 *  Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


START_NAMESPACE_STIR

Succeeded
InputStreamWithRecordsFromUPENNtxt::
create_output_file(std::string ofilename)
{
    error("InputStreamWithRecordsFromUPENNtxt: We do not support this here!");
}


END_NAMESPACE_STIR
