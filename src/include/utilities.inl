//
// $Id$: $Date$
//
/*!
  \file 
  \ingroup buildblock 
  \brief inline implementations for utility.h

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$
*/
#include <iostream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
#endif

START_NAMESPACE_TOMO
/*!
  The question is currently presented as
  \verbatim
  str_text : [minimum_value, maximum_value, D: default_value]: 
  \endverbatim
  Simply pressing 'enter' will select the default value. Otherwise, range 
  checking is performed, and the question asked again if necessary.
*/
template <class NUMBER>
inline NUMBER 
ask_num (const string& str,
	 NUMBER minimum_value,
	 NUMBER maximum_value,
	 NUMBER default_value)
{ 
  
  while(1)
  { 
    char input[30];

    cerr << "\n" << str 
         << "[" << minimum_value << "," << maximum_value 
	 << " D:" << default_value << "]: ";
    fgets(input,30,stdin);
#ifdef BOOST_NO_STRINGSTREAM
    istrstream ss(input);
#else
    std::istringstream ss(input);
#endif
    
    NUMBER value = default_value;
    ss >> value;
    if ((value>=minimum_value) && (maximum_value>=value))
      return value;
    cerr << "\nOut of bounds. Try again.";
  }
}



template <class IFSTREAM>
inline IFSTREAM& open_read_binary(IFSTREAM& s, 
				  const char * const name)
{
#if 0
  //KT 30/07/98 The next lines are only necessary (in VC 5.0) when importing 
  // <fstream.h>. We use <fstream> now, so they are disabled.

  // Visual C++ does not complain when opening a nonexisting file for reading,
  // unless using ios::nocreate
  s.open(name, ios::in | ios::binary | ios::nocreate); 
#else
  s.open(name, ios::in | ios::binary); 
#endif
  // KT 14/01/2000 added name of file in error message
  if (s.fail() || s.bad())
    { error("Error opening file %s\n", name);  }
  return s;
}

template <class OFSTREAM>
inline OFSTREAM& open_write_binary(OFSTREAM& s, 
				  const char * const name)
{
    s.open(name, ios::out | ios::binary); 
    // KT 14/01/2000 added name of file in error message
    if (s.fail() || s.bad())
    { error("Error opening file %s\n", name); }
    return s;
}



#ifndef _MSC_VER
char *strupr(char * const str)
{
  for (char *a = str; *a; a++)
  {
    if ((*a >= 'a')&&(*a <= 'z')) *a += 'A'-'a';
  };
  return str;
}
#endif

END_NAMESPACE_TOMO
