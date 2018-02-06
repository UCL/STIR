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

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
  */
#ifndef __stir_IO_ecat6_types_h
#define __stir_IO_ecat6_types_h

// include this for namespace macros
#include "stir/IO/stir_ecat_common.h"
// define to use the original version of the code
//#define STIR_ORIGINAL_ECAT6

#ifndef STIR_ORIGINAL_ECAT6
#ifdef STIR_NO_NAMESPACES
// terrible trick to avoid conflict between stir::Sinogram and Sinogram defined in matrix.h
// when we do have namespaces, the conflict can be resolved by using ::Sinogram
#define Sinogram CTISinogram
#else
#define CTISinogram ::Sinogram
#endif

#include "matrix.h"

#ifdef STIR_NO_NAMESPACES
#undef Sinogram
#endif

#endif // STIR_ORIGINAL_ECAT6

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6

// matrix.h defines MatBLKSIZE and MatFirstDirBlk. we undefine them here to avoid conflicts
#ifdef MatBLKSIZE
#undef MatBLKSIZE
#endif
#ifdef MatFirstDirBlk
#undef MatFirstDirBlk
#endif
const int MatBLKSIZE =          512;
const int MatFirstDirBlk =       2;

// MatFileType
typedef enum {
    matScanFile = 1,   // sinogram data
    matImageFile = 2,  // image file   
    matAttenFile = 3,  // attenuation correction file
    matNormFile = 4    // normalization file
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
typedef struct ScanInfoRec {
    int nprojs,
        nviews,
        nblks,
        strtblk;
    word data_type;
} ScanInfoRec;

#ifndef STIR_ORIGINAL_ECAT6
typedef Main_header ECAT6_Main_header;
typedef struct Matval Matval;

#else // STIR_ORIGINAL_ECAT6

#error STIR_ORIGINAL_ECAT6 no longer supported

#endif // STIR_ORIGINAL_ECAT6
  
END_NAMESPACE_ECAT
END_NAMESPACE_ECAT6
END_NAMESPACE_STIR
#endif 
