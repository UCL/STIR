//
// $Id$: $Date$
//

#ifndef __Tomography_common_H__
#define __Tomography_common_H__

/*!
  \file 
 
  \brief basic configuration include file 

  \author Kris Thielemans
  \author Alexey Zverovich
  \author Darren Hague
  \author PARAPET project

  \date    $Date$

  \version $Revision$



 This include file defines some commonly used macros, templates 
 and functions in an attempt to smooth out some system dependencies.
 It also defines some functions which are used very often.

 \par Macros and system dependencies:

 - macros for namespace support: 
   #defines TOMO_NO_NAMESPACES if the compiler does not support namespaces
   #defines START_NAMESPACE_TOMO etc.

 - includes boost/config.hpp

 - #defines TOMO_NO_COVARIANT_RETURN_TYPES when the compiler does not
   support virtual functions of a derived class differing only in the return
   type.
   
 - preprocessor definitions which attempt to determine the 
   operating system this is going to run on.
   use as #ifdef  __OS_WIN__ ... #elif ... #endif
   Possible values are __OS_WIN__, __OS_MAC__, __OS_VAX__, __OS_UNIX__
   The __OS_UNIX__ case has 'subbranches'. At the moment we attempt to find
   out on __OS_AIX__, __OS_SUN__, __OS_OSF__, __OS_LINUX__.
   (If the attempts fail to determine the correct OS, you can pass
    the correct value as a preprocessor definition to the compiler)
 
 - #includes cstdio, cstdlib, cstring, cmath

 - templates const T& std::min(const T& x, const T& y) and std::max (if not provided)
   (source files should still include <algorithm> though)

 - general definitions of operator !=, >, <= and >= in terms of == and < (if not provided)
 
 - a feable attempt to be able to use the old strstream and 
   the new stringstream classes in the same way

 - a trick to get ANSI C++ 'for' scoping rules work, even for compilers
   which do not follow the new standard

 - #ifdef TOMO_ASSERT, then define our own assert, else include <cassert>


\par Speeding up std::copy

 - overloads of std::copy for built-in types to use memcpy (so it's faster)


\par Tomography namespace members declared here
  
 - const double _PI
 
 - error(const char * const format_string, ...)
   writes error information a la printf.

 - inline template <class NUMBER> NUMBER square(const NUMBER &x)

 */

#ifdef _MSC_VER
// disable warnings on very long identifiers for debugging information
#pragma warning(disable: 4786)

#endif // _MSC_VER

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "boost/config.hpp"

//*************** preprocessor definitions for old compilers

//**** namespace support

#if defined __GNUC__
# if __GNUC__ == 2 && __GNUC_MINOR__ <= 8
#  define TOMO_NO_NAMESPACES
# endif
#endif

//**** TOMO_NO_COVARIANT_RETURN_TYPES
/* Define when your compiler does not handle the following:
   class A { virtual A* f();}
   class B:A { virtual B* f(); }
*/
#ifdef _MSC_VER
#define TOMO_NO_COVARIANT_RETURN_TYPES
#endif


//*************** namespace macros

#ifndef TOMO_NO_NAMESPACES
//TODO remove
#ifdef TOMO
# define START_NAMESPACE_TOMO namespace Tomography {
# define END_NAMESPACE_TOMO }
# define USING_NAMESPACE_TOMO using namespace Tomography;
#else
# define USING_NAMESPACE_TOMO 
# define START_NAMESPACE_TOMO 
# define END_NAMESPACE_TOMO 
#endif
# define START_NAMESPACE_STD namespace std {
# define END_NAMESPACE_STD }
# define USING_NAMESPACE_STD using namespace std;
#else
# define START_NAMESPACE_TOMO 
# define END_NAMESPACE_TOMO 
# define USING_NAMESPACE_TOMO 
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

//*************** min, max
// STL should have min,max in <algorithm>, 
// but vanilla VC++ has a conflict between std::min and some preprocessor defs
#if defined (_MSC_VER) && !defined(__STL_CONFIG_H)
#undef min
#undef max

namespace std {
template <class T>
inline const T& min(const T& a, const T& b) 
{
  return b < a ? b : a;
}

template <class T>
inline const T& max(const T& a, const T& b) 
{
  return  a < b ? b : a;
}

}
#endif // min,max defs


//*************** !=, >, <= and >= 

#if defined (_MSC_VER) && !defined(__STL_CONFIG_H)
// general definitions of operator !=, >, <= and >= in terms of == and <
// (copied from SGI stl_relops.h)
template <class T>
inline bool operator!=(const T& x, const T& y) {
  return !(x == y);
}

template <class T>
inline bool operator>(const T& x, const T& y) {
  return y < x;
}

template <class T>
inline bool operator<=(const T& x, const T& y) {
  return !(y < x);
}

template <class T>
inline bool operator>=(const T& x, const T& y) {
  return !(x < y);
}

#endif // !=,>,<=,>=

//*************** overload std::copy for built-in types

#include <algorithm>

START_NAMESPACE_STD
template <>
inline double * 
copy(const double * first, const double * last, double * to)
{  memcpy(to, first, (last-first)*sizeof(double)); return to+(last-first); }

template <>
inline float * 
copy(const float * first, const float * last, float * to)
{  memcpy(to, first, (last-first)*sizeof(float)); return to+(last-first); }

template <>
inline unsigned long int * 
copy(const unsigned long int * first, const unsigned long int * last, unsigned long int * to)
{  memcpy(to, first, (last-first)*sizeof(unsigned long int)); return to+(last-first); }

template <>
inline signed long int * 
copy(const signed long int * first, const signed long int * last, signed long int * to)
{  memcpy(to, first, (last-first)*sizeof(signed long int)); return to+(last-first); }

template <>
inline unsigned int * 
copy(const unsigned int * first, const unsigned int * last, unsigned int * to)
{  memcpy(to, first, (last-first)*sizeof(unsigned int)); return to+(last-first); }

template <>
inline signed int * 
copy(const signed int * first, const signed int * last, signed int * to)
{  memcpy(to, first, (last-first)*sizeof(signed int)); return to+(last-first); }

template <>
inline unsigned short int * 
copy(const unsigned short int * first, const unsigned short int * last, unsigned short int * to)
{  memcpy(to, first, (last-first)*sizeof(unsigned short int)); return to+(last-first); }

template <>
inline signed short int * 
copy(const signed short int * first, const signed short int * last, signed short int * to)
{  memcpy(to, first, (last-first)*sizeof(signed short int)); return to+(last-first); }

template <>
inline unsigned char * 
copy(const unsigned char * first, const unsigned char * last, unsigned char * to)
{  memcpy(to, first, (last-first)*sizeof(unsigned char)); return to+(last-first); }

template <>
inline signed char * 
copy(const signed char * first, const signed char * last, signed char * to)
{  memcpy(to, first, (last-first)*sizeof(signed char)); return to+(last-first); }

template <>
inline char * 
copy(const char * first, const char * last, char * to)
{  memcpy(to, first, (last-first)*sizeof(char)); return to+(last-first); }


template <>
inline bool * 
copy(const bool * first, const bool * last, bool * to)
{  memcpy(to, first, (last-first)*sizeof(bool)); return to+(last-first); }

END_NAMESPACE_STD


//*************** strstream stuff

// some trickery to make strstreams work most of the time. 
// gcc 2.95.2 has still only the old strstream libraries.
// TODO These macros should really work the other way around: old in terms of new
#if defined(_MSC_VER) || defined(__MSL__)
#  include <sstream>
#  define strstream stringstream
#  define istrstream istringstream
#  define ostrstream ostringstream
#else
#  include <strstream>
#endif

//*************** for() scope trick

/* ANSI C++ (re)defines the scope of variables declared in a for() statement.
   Example: the 'i' variable has scope only within the for statement.
   for (int i=0; ...)
     do_something;
   The next trick (by AZ) solves this problem.
   At the moment, we only need it for VC++ 
   */
   
#ifdef _MSC_VER
#	ifndef for
#		define for if (0) ; else for
#	else
#		error 'for' is already #defined to something 
#	endif
#endif

//*************** assert

#ifndef TOMO_ASSERT
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
#endif // TOMO_ASSERT

//*************** 
START_NAMESPACE_TOMO

//! The constant pi to high precision.
const double _PI = 3.14159265358979323846264338327950288419716939937510;

//! Print error with format string a la \c printf and abort
void error(const char *const s, ...);

//! Print warning with format string a la \c printf
void warning(const char *const s, ...);

//! returns the square of a number, templated.
template <class NUMBER> 
inline NUMBER square(const NUMBER &x) { return x*x; }


END_NAMESPACE_TOMO

#endif 
