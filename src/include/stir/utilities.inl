/*!
  \file 
  \ingroup buildblock 
  \brief inline implementations for utility.h

  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
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
#include <iostream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

START_NAMESPACE_STIR
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
ask_num (const std::string& str,
	 NUMBER minimum_value,
	 NUMBER maximum_value,
	 NUMBER default_value)
{ 
  
  while(1)
  { 
    std::string input;
    std::cerr << "\n" << str 
         << "[" << minimum_value << "," << maximum_value 
	 << " D:" << default_value << "]: ";
    std::getline(std::cin, input);
#ifdef BOOST_NO_STRINGSTREAM
    istrstream ss(input.c_str());
#else
    std::istringstream ss(input.c_str());
#endif
    
    NUMBER value = default_value;
    ss >> value;
    if ((value>=minimum_value) && (maximum_value>=value))
      return value;
    std::cerr << "\nOut of bounds. Try again.";
  }
}



template <class IFSTREAM>
inline IFSTREAM& open_read_binary(IFSTREAM& s, 
				  const std::string& name)
{
#if 0
  //KT 30/07/98 The next lines are only necessary (in VC 5.0) when importing 
  // <fstream.h>. We use <fstream> now, so they are disabled.

  // Visual C++ does not complain when opening a nonexisting file for reading,
  // unless using std::ios::nocreate
  s.open(name.c_str(), std::ios::in | std::ios::binary | std::ios::nocreate); 
#else
  s.open(name.c_str(), std::ios::in | std::ios::binary); 
#endif
  // KT 14/01/2000 added name of file in error message
  if (s.fail() || s.bad())
    { error("Error opening file %s\n", name.c_str());  }
  return s;
}

template <class OFSTREAM>
inline OFSTREAM& open_write_binary(OFSTREAM& s, 
				  const std::string& name)
{
    s.open(name.c_str(), std::ios::out | std::ios::binary); 
    // KT 14/01/2000 added name of file in error message
    if (s.fail() || s.bad())
    { error("Error opening file %s\n", name.c_str()); }
    return s;
}

template <class FSTREAM>
inline void close_file(FSTREAM& s)
{
  s.close();
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

END_NAMESPACE_STIR
