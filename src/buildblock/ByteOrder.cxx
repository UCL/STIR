//
// $Id$: $Date$
//

/* 
  Initialisation of the ByteOrder static member.

  History:
   first version Kris Thielemans

   KT&AZ 08/12/99 avoid using ntohs()
*/

#include "ByteOrder.h"

#if 1

// KT&AZ 08/12/99 new 
/* A somewhat complicated way to determine the byteorder.
   The advantage is that it doesn't need ntohs (and so any
   compiler specific definitions, libraries or whatever).
   Philosophy: initialise a long number, look at its first byte
   by casting its address as a char *.
   The reinterpret_cash is to make it typesafe for C++.

   First we do a (paranoid) check : 
      sizeof(unsigned long) - sizeof(unsigned char) > 0
   This is done via a 'compile-time assertion', i.e. it breaks 
   at compile time when the assertion is false. The line below
   relies on the fact that you cannot have an array with
   zero (or less) elements.
 */

typedef char 
  assert_unsigned_long_size[sizeof(unsigned long) - sizeof(unsigned char)];

static const unsigned long magic = 1;

const ByteOrder::Order ByteOrder::native_order =
  *(reinterpret_cast<const unsigned char*>(&magic) ) == 1 ?
      little_endian : big_endian;

#else
// old implementation of byte-order stuff
// this uses ntohs, which is standard on Unix, but you have to fiddle around on 
// NT systems

// includes for ntohs
#ifdef _WIN32

#ifdef __GNUG__ // Cygwin32
#include <asm/byteorder.h>
#else           // Other Windows compilers
// Unfortunately, the definition of ntohs is in an Import Library 
// Windows NT, Windows 95: ws2_32.lib; Win32s: wsock32.lib 
// The relevant library has to be added to your project. 
#include <winsock2.h>
#endif

#else // !_WIN32

#include <sys/types.h>
#include <netinet/in.h>

#endif

#if defined(_WIN32) && defined (__GNUG__)
// Cygwin32 does define ntohs(), but it's in some library which
// KT couldn not locate. However, __ntohs is defined as inline asm.
// so we just use that one.

// KT 30/10/98 corrected bug: it returned the other byte order
const ByteOrder::Order ByteOrder::native_order = 
   (__ntohs(1) != 1) ? little_endian : big_endian;

#else // !Cygwin32

const ByteOrder::Order ByteOrder::native_order = 
   (ntohs(1) != 1) ? little_endian : big_endian;

#endif // Cygwin32

#endif // old implementation of byte order stuff
