//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Definition of STIRByteOrderIsBigEndian
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/common.h"

START_NAMESPACE_STIR

#ifndef DOXYGEN_SKIP
// define both if them when doxygen is run
// otherwise doxygen will only generate the doc for the one that is actually defined

#define STIRNativeByteOrderIsBigEndian
#define STIRNativeByteOrderIsLittleEndian
#endif

/*!
  \def  STIRNativeByteOrderIsBigEndian
  \ingroup buildblock  
  \brief A macro that is defined \e only when the compilation is on
  a big endian machine.

  This should be used only in the case that you really need to know this
  at compilation time. Try to use class ByteOrder instead.

  The definition of class ByteOrder is independent of this macro.

  \warning It is recommended to check these settings by using an assert on 
  ByteOrder.
  \warning This macro depends on preprocessor defines set by your compiler.
  The list of defines is bound to be incomplete though. The current list sets
  LittleEndian for alpha, x86 or PowerPC processors or Windows using some
  macros which are known to work with gcc and Visual Studio (at some point also
  with CodeWarrior). Otherwise, the
  architecture is assumed to be big endian. ByteOrder does not rely on 
  this type of conditionals.
*/

/*!
  \def  STIRNativeByteOrderIsLittleEndian
  \ingroup buildblock  
  \brief A macro that is defined \e only when the compilation is on
  a little endian machine.

  \see STIRNativeByteOrderIsBigEndian for details.
*/

// currently checked by asserts()
#if !defined(__alpha) && (!defined(_WIN32) || defined(_M_PPC) || defined(_M_MPPC)) && !defined(__i386__) && !defined(__i486__) && !defined(__i586__) && !defined(__i686__) && !defined(__i786__)&& !defined(__i886__) && !defined(__k6__) && !defined(__athlon__) || (defined(__MSL__) && !defined(__LITTLE_ENDIAN))
#define STIRNativeByteOrderIsBigEndian 1
#else
#define STIRNativeByteOrderIsLittleEndian 1
#endif
END_NAMESPACE_STIR

