//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

	See STIR/LICENSE.txt for details
*/

#ifndef __stir_config_gcc_H__
#define __stir_config_gcc_H__

/*!
  \file 
  \ingroup buildblock 
  \brief configuration for gcc

  \author Kris Thielemans
  \author PARAPET project




 This include file defines a few macros and en/disables pragmas
 specific to gcc.

 It is included by sitr/common.h. You should never include it directly.
*/

#if defined __GNUC__
# if __GNUC__ == 2 && __GNUC_MINOR__ <= 8
#  define STIR_NO_NAMESPACES
#  define STIR_NO_AUTO_PTR
# endif
#endif

#endif 
