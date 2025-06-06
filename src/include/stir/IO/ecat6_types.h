//
//

/*!
  \file
  \ingroup ECAT
  \brief ECAT 6 CTI matrix parameters
  \author Larry Byars
  \author PARAPET project

  Enumerations of the data type and format.

  Structures: <BR>
  - scanner parameters <BR>
  - matrix blocks and parameters <BR>
  - main header, scan and image subheaders
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
  */
#ifndef __stir_IO_ecat6_types_h
#define __stir_IO_ecat6_types_h

// include this for namespace macros
#include "stir/IO/stir_ecat_common.h"
// define to use the original version of the code
//#define STIR_ORIGINAL_ECAT6

#ifndef STIR_ORIGINAL_ECAT6
#  define CTISinogram ::Sinogram

#  include "matrix.h"

#endif // STIR_ORIGINAL_ECAT6

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6

// matrix.h defines MatBLKSIZE and MatFirstDirBlk. we undefine them here to avoid conflicts
#ifdef MatBLKSIZE
#  undef MatBLKSIZE
#endif
#ifdef MatFirstDirBlk
#  undef MatFirstDirBlk
#endif
const int MatBLKSIZE = 512;
const int MatFirstDirBlk = 2;

// MatFileType
typedef enum
{
  matScanFile = 1,  // sinogram data
  matImageFile = 2, // image file
  matAttenFile = 3, // attenuation correction file
  matNormFile = 4   // normalization file
} MatFileType;

//#define matScanData   matI2Data
//#define matImageData  matI2Data

typedef short word;
typedef unsigned char byte;

/*!
  \struct ScanInfoRec
  \brief ECAT 6 CTI scanner parameters
  \ingroup ECAT
  \param nprojs      number of projections
  \param nviews      number of views
  \param nblks       number of blocks (planes)
  \param strtblk     first block
  \param data_type   type of data (float, short, ...)
*/
typedef struct ScanInfoRec
{
  int nprojs, nviews, nblks, strtblk;
  word data_type;
} ScanInfoRec;

#ifndef STIR_ORIGINAL_ECAT6
typedef Main_header ECAT6_Main_header;
typedef struct Matval Matval;

#else // STIR_ORIGINAL_ECAT6

#  error STIR_ORIGINAL_ECAT6 no longer supported

#endif // STIR_ORIGINAL_ECAT6

END_NAMESPACE_ECAT
END_NAMESPACE_ECAT6
END_NAMESPACE_STIR
#endif
