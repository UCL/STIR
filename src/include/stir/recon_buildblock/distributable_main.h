//
//
/*
    Copyright (C) 2011- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_DISTRIBUTABLE_main_H__
#define __stir_recon_buildblock_DISTRIBUTABLE_main_H__

/*!
  \file
  \ingroup distributable

  \brief Declaration of the stir::distributable_main function

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project
*/
#include "stir/common.h"

START_NAMESPACE_STIR

//!@{
//! \ingroup distributable

//! main function that starts processing on the master or slave as appropriate
/*! This function should be implemented by the application, replacing the usual ::main().

    DistributedWorker.cxx provides a ::main() function that will set-up everything for
    parallel processing (as appropriate on the master and slaves), and then calls
    distributable_main() on the master (passing any arguments along).

    If STIR_MPI is not defined, ::main() will simply call distributable_main().

    Skeleton of a program that uses this module:
    \code
    #include "stir/recon_buildblock/distributable_main.h"

    int stir::distributable_main(int argc, char** argv)
    {
    // master node code goes here
    }
    \endcode

*/
int distributable_main(int argc, char** argv);

//!@}

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_DISTRIBUTABLE_main_H__
