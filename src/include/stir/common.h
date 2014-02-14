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
#ifndef __stir_common_H__
#define __stir_common_H__

/*!
  \file 
  \ingroup buildblock 
  \brief basic configuration include file 

  \author Kris Thielemans
  \author Alexey Zverovich
  \author Darren Hague
  \author PARAPET project





 This include file defines some commonly used macros, templates 
 and functions in an attempt to smooth out some system dependencies.
 It also defines some functions which are used very often.

 <H3> Macros and system dependencies:</H3>

<UL>
 <LI> macros for namespace support: 
   #defines STIR_NO_NAMESPACES if the compiler does not support namespaces
   #defines START_NAMESPACE_STIR etc.

 <LI> includes boost/config.hpp

 <LI> #defines STIR_NO_COVARIANT_RETURN_TYPES when the compiler does not
   support virtual functions of a derived class differing only in the return
   type.
   Define when your compiler does not handle the following:
   \code
   class A { virtual A* f();}
   class B:A { virtual B* f(); }
   \endcode

 <LI> #defines STIR_NO_AUTO_PTR when the compiles has no std::auto_ptr support.
 In that case, we #define auto_ptr to shared_ptr
   
 <LI> preprocessor definitions which attempt to determine the 
   operating system this is going to run on.
   use as #ifdef  __OS_WIN__ ... #elif ... #endif
   Possible values are __OS_WIN__, __OS_MAC__, __OS_VAX__, __OS_UNIX__
   The __OS_UNIX__ case has 'subbranches'. At the moment we attempt to find
   out on __OS_AIX__, __OS_SUN__, __OS_OSF__, __OS_LINUX__.
   (If the attempts fail to determine the correct OS, you can pass
    the correct value as a preprocessor definition to the compiler)
 
 <LI> #includes cstdio, cstdlib, cstring, cmath

 <LI> a trick to get ANSI C++ 'for' scoping rules work for compilers
   which do not follow the new standard

 <LI> #ifdef STIR_ASSERT, then define our own assert, else include &lt;cassert&gt;

</UL>

<H3> Speeding up std::copy</H3>

 <UL>
 <LI> For old compilers (check the source!), overloads of std::copy for built-in 
      types to use memmove (so it's faster)
 </UL>

<H3> stir namespace members declared here</H3>
  
 <UL>
 <LI> <tt>const double _PI</tt></li>

 <LI> <tt>inline template  &lt;class NUMBER&gt; NUMBER square(const NUMBER &x) </tt></li>

 </UL>

<h3> stir include files included here</H3>
  <UL>
  <li> <tt>stir/error.h</tt> definiong stir::error</li>
  <li> <tt>stir/warning.h</tt> definiong stir::warning</li>
  </UL>
*/
#if defined(_MSC_VER)
#include "stir/config/visualc.h"
#endif
#if defined(__GNUC__)
#include "stir/config/gcc.h"
#endif

#include "boost/config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>


//*************** namespace macros
#ifndef STIR_NO_NAMESPACES

# define START_NAMESPACE_STIR namespace stir {
# define END_NAMESPACE_STIR }
# define USING_NAMESPACE_STIR using namespace stir;
# define START_NAMESPACE_STD namespace std {
# define END_NAMESPACE_STD }
# define USING_NAMESPACE_STD using namespace std;
#else
# define START_NAMESPACE_STIR 
# define END_NAMESPACE_STIR 
# define USING_NAMESPACE_STIR 
# define START_NAMESPACE_STD
# define END_NAMESPACE_STD 
# define USING_NAMESPACE_STD 
#endif


//*************** define __OS_xxx__

#if !defined(__OS_WIN__) && !defined(__OS_MAC__) && !defined(__OS_VAX__) && !defined(__OS_UNIX__) 
// if none of these macros is defined externally, we attempt to guess, defaulting to UNIX

#ifdef __MSL__
   // Metrowerks CodeWarrior
   // first set its own macro
#  if macintosh && !defined(__dest_os)
#    define __dest_os __mac_os
#  endif
#  if __dest_os == __mac_os
#    define __OS_MAC__
#  else
#    define __OS_WIN__
#  endif

#elif defined(_WIN32) || defined(WIN32) || defined(_WINDOWS) || defined(_DOS)
  // Visual C++, MSC, cygwin gcc and hopefully some others
# define __OS_WIN__

#elif defined(VAX)
   // Just in case anyone is still using VAXes...
#  define __OS_VAX__

#else // default

#  define __OS_UNIX__
   // subcases
#  if defined(_AIX)
#    define __OS_AIX__
#  elif defined(__sun)
     // should really branch on SunOS and Solaris...
#    define __OS_SUN__
#  elif defined(__linux__)
#    define __OS_LINUX__
#  elif defined(__osf__)
#    defined __OS_OSF__
#  endif

#endif  // __OS_UNIX_ case

#endif // !defined(__OS_xxx_)

//************** auto_ptr
#if defined(BOOST_NO_AUTO_PTR) && !defined(STIR_NO_AUTO_PTR)
#  define STIR_NO_AUTO_PTR
#endif

#ifdef STIR_NO_AUTO_PTR
// first include memory, just in case there is a (supposedly flawed) auto_ptr in there
#include <memory>
// now include our own shared_ptr (somewhat tricky, as this will 
// include this file again...)
#include "stir/shared_ptr.h"
#define auto_ptr shared_ptr
#endif


//*************** overload std::copy for built-in types
/* If you have an older compiler, chages are that std::copy is 
   implemented in the obvious way of iterating and copying along 
   the way. However, for simple types (such as floats), calling 
   memmove (not memcpy as the ranges could overlap) is faster.
   So, we overload std::copy for some built-in types.
  
   However, newer compilers (in particular gcc from version 2.8)
   take care of this themselves. So, we only do this 
   conditionally.
*/
#ifdef STIR_SPEED_UP_STD_COPY
#include <algorithm>

START_NAMESPACE_STD
//! overloads std::copy for faster performance
template <>
inline double * 
copy(const double * first, const double * last, double * to)
{  memmove(to, first, (last-first)*sizeof(double)); return to+(last-first); }

template <>
inline float * 
copy(const float * first, const float * last, float * to)
{  memmove(to, first, (last-first)*sizeof(float)); return to+(last-first); }

template <>
inline unsigned long int * 
copy(const unsigned long int * first, const unsigned long int * last, unsigned long int * to)
{  memmove(to, first, (last-first)*sizeof(unsigned long int)); return to+(last-first); }

template <>
inline signed long int * 
copy(const signed long int * first, const signed long int * last, signed long int * to)
{  memmove(to, first, (last-first)*sizeof(signed long int)); return to+(last-first); }

template <>
inline unsigned int * 
copy(const unsigned int * first, const unsigned int * last, unsigned int * to)
{  memmove(to, first, (last-first)*sizeof(unsigned int)); return to+(last-first); }

template <>
inline signed int * 
copy(const signed int * first, const signed int * last, signed int * to)
{  memmove(to, first, (last-first)*sizeof(signed int)); return to+(last-first); }

template <>
inline unsigned short int * 
copy(const unsigned short int * first, const unsigned short int * last, unsigned short int * to)
{  memmove(to, first, (last-first)*sizeof(unsigned short int)); return to+(last-first); }

template <>
inline signed short int * 
copy(const signed short int * first, const signed short int * last, signed short int * to)
{  memmove(to, first, (last-first)*sizeof(signed short int)); return to+(last-first); }

template <>
inline unsigned char * 
copy(const unsigned char * first, const unsigned char * last, unsigned char * to)
{  memmove(to, first, (last-first)*sizeof(unsigned char)); return to+(last-first); }

template <>
inline signed char * 
copy(const signed char * first, const signed char * last, signed char * to)
{  memmove(to, first, (last-first)*sizeof(signed char)); return to+(last-first); }

template <>
inline char * 
copy(const char * first, const char * last, char * to)
{  memmove(to, first, (last-first)*sizeof(char)); return to+(last-first); }


template <>
inline bool * 
copy(const bool * first, const bool * last, bool * to)
{  memmove(to, first, (last-first)*sizeof(bool)); return to+(last-first); }

END_NAMESPACE_STD

#endif // #ifdef STIR_SPEED_UP_STD_COPY

//*************** for() scope trick

/* ANSI C++ (re)defines the scope of variables declared in a for() statement.
   Example: the 'i' variable has scope only within the for statement.
   for (int i=0; ...)
     do_something;
   The next trick (by AZ) solves this problem.
   At the moment, we only need it for VC++ 
   */
   
#if defined(STIR_ENABLE_FOR_SCOPE_WORKAROUND)
#	ifndef for
#		define for if (0) ; else for
#	else
#		error 'for' is already #defined to something 
#	endif
#endif

//*************** assert

#ifndef STIR_ASSERT
#  include <cassert>
#else
  // use our own assert
#  ifdef assert
#    undef assert
#  endif
#  if !defined(NDEBUG)
#    define assert(x) {if (!(x)) { \
      fprintf(stderr,"Assertion \"%s\" failed in file %s:%d\n", # x,__FILE__, __LINE__); \
      abort();} }
#  else 
#     define assert(x)
#  endif
#endif // STIR_ASSERT

//*************** 
START_NAMESPACE_STIR

//! The constant pi to high precision.
/*! \ingroup buildblock */
const double _PI = 3.14159265358979323846264338327950288419716939937510;

//! returns the square of a number, templated.
/*! \ingroup buildblock */
template <class NUMBER> 
inline NUMBER square(const NUMBER &x) { return x*x; }


END_NAMESPACE_STIR

// include these such that we don't have to include them all over the place
#include "stir/error.h"
#include "stir/warning.h"


#endif 
