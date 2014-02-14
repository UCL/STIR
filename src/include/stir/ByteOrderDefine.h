//
//
/*!

  \file
  \ingroup buildblock  
  \brief Definition of STIRIsNativeByteOrderBigEndian and STIRIsNativeByteOrderLittleEndian preprocessor symbols
    
  \author Kris Thielemans
      
*/
/*
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

#include "stir/common.h"

START_NAMESPACE_STIR

// currently checked by asserts()
#if !defined(__alpha) && (!defined(_WIN32) || defined(_M_PPC) || defined(_M_MPPC)) && !defined(__i386__) && !defined(__i486__) && !defined(__i586__) && !defined(__i686__) && !defined(__i786__)&& !defined(__i886__) && !defined(__k6__) && !defined(__athlon__) && !defined(__x86_64__) && !defined(__k6__) || (defined(__MSL__) && !defined(__LITTLE_ENDIAN))
#define STIRIsNativeByteOrderBigEndian 1
#define STIRIsNativeByteOrderLittleEndian 0
#else
#define STIRIsNativeByteOrderBigEndian 0
#define STIRIsNativeByteOrderLittleEndian 1
#endif

/*!
  \def  STIRIsNativeByteOrderBigEndian
  \ingroup buildblock  
  \brief A macro that is defined to 1 when the compilation is on
  a big endian machine, otherwise it is set to 0. 

  \par Usage
  \code
  #if STIRIsNativeByteOrderBigEndian
  // code specific for big endian machines
  #else
  // code for little endian machines
  #endif
  \endcode

  \par Relation to the ByteOrder class.

  This should be used only in the case that you really need to know this
  at compilation time. Try to use class ByteOrder instead.

  The definition of class ByteOrder is independent of this macro.

  \warning It is recommended to check these settings by comparing with class 
  ByteOrder. This is done by the ByteOrderTests class.

  \warning This macro depends on preprocessor defines set by your compiler.
  The list of defines is bound to be incomplete though. The current list sets
  LittleEndian for alpha, x86 or PowerPC processors or Windows using some
  macros which are known to work with gcc and Visual Studio (at some point also
  with CodeWarrior). Otherwise, the
  architecture is assumed to be big endian. ByteOrder does not rely on 
  this type of conditionals.
  \warning   The value of the macro in the doxygen documentation depends on 
  what type of machine doxygen was run.

*/

/*!
  \def  STIRIsNativeByteOrderLittleEndian
  \ingroup buildblock  
  \brief A macro that is defined to 1 when the compilation is on
  a little endian machine, and to 0 otherwise.

  \see STIRIsNativeByteOrderBigEndian for details.
*/


END_NAMESPACE_STIR

