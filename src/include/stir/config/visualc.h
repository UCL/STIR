//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2010, Hammersmith Imanet Ltd
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

#ifndef __stir_config_visualc_H__
#define __stir_config_visualc_H__

/*!
  \file 
  \ingroup buildblock 
  \brief configuration for Visual C++

  \author Kris Thielemans
  \author PARAPET project




 This include file defines a few macros and en/disables pragmas
 specific to Microsoft Visual C++.

 It is included by sitr/common.h. You should never include it directly.
*/

#if defined(_MSC_VER) && _MSC_VER<=1300
// do this only up to VC 7.0
#define STIR_NO_COVARIANT_RETURN_TYPES
#define STIR_SPEED_UP_STD_COPY
#define STIR_ENABLE_FOR_SCOPE_WORKAROUND
#endif

#if defined(_MSC_VER)
// set _SCL_SECURE_NO_WARNINGS
// otherwise we get a load of messages that std::copy and std::equal are unsafe 
// in VectorWithOffset and IndexRange etc because they use C-style arrays internally
#pragma warning( disable : 4996) 

// enable secure versions of standard C functions such as sprintf etc
// this will cause a run-time error when overwriting memory etc
// hopefully this is enough to avoid a lot of warnings
// otherwise we'll need to set define _CTR_SECURE_NO_WARNINGS
#ifdef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
  // it's already defined. let's get rid of it.
# undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#endif
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif 

#endif
