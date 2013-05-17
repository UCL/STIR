//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Input/output of basic vector-like types to/from streams

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009 Hammersmith Imanet Ltd
    Copyright (C) 2013 Kris Thielemans
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
/* History:
   KT 08/12/2000 corrected cases in operator<< for 0 length
   KT 29/08/2001 added operator>>
   KT 07/10/2002 corrected case in operator>> for 0 length and added more checking on stream state
*/

#include <algorithm>

START_NAMESPACE_STIR

template <typename elemT>
std::ostream& 
operator<<(std::ostream& str, const VectorWithOffset<elemT>& v)
{
      str << '{';
      for (int i=v.get_min_index(); i<v.get_max_index(); i++)
	str << v[i] << ", ";

      if (v.get_length()>0)
	str << v[v.get_max_index()];
      str << '}' << std::endl;
      return str;
}

template <int num_dimensions, typename coordT>
std::ostream& 
operator<<(std::ostream& str, const BasicCoordinate<num_dimensions, coordT>& v)
{
      str << '{';
      for (int i=1; i<num_dimensions; i++)
	str << v[i] << ", ";
      
      if (num_dimensions>0)
	str << v[num_dimensions];
      str << '}';
      return str;
}


template <typename elemT>
std::ostream& 
operator<<(std::ostream& str, const std::vector<elemT>& v)
{
      str << '{';
      // slightly different from above because vector::size() is unsigned
      // so 0-1 == 0xFFFFFFFF (and not -1)
      if (v.size()>0)
      {
        for (unsigned int i=0; i<v.size()-1; i++)
	  str << v[i] << ", ";      
	str << v[v.size()-1];
      }
      str << '}' << std::endl;
      return str;
}

template <typename elemT>
std::istream& 
operator>>(std::istream& str, std::vector<elemT>& v)
{
  v.resize(0);
  char c;
  str >> std::ws >> c;
  if (!str || c != '{')
    return str;
  
  elemT t;
  do
  {
    str >> t;
    if (!str.fail())
    { 
      v.push_back(t);
      str >> std::ws >> c;
    }
    else 
      break;
  }
  while (str && c  == ',');

  if (str.fail())
  {
    str.clear();
    str >> std::ws >> c;
  }
  if (!str)
  {
    warning("\nreading a vector, expected closing }, but found EOF or worse. Length of vector returned is %ud\n", v.size());
    return str;
  }
    
  if (c!= '}')
    warning("\nreading a vector, expected closing }, found %c instead. Length of vector returned is %u\n", c, v.size());
  return str;
}

template <typename elemT>
std::istream& 
operator>>(std::istream& str, VectorWithOffset<elemT>& v)
{
  std::vector<elemT> vv;
  str >> vv;
  v = VectorWithOffset<elemT>(static_cast<int>(vv.size()));
  std::copy(vv.begin(), vv.end(), v.begin());
  return str;
}

template <int num_dimensions, typename coordT>
std::istream& 
operator>>(std::istream& str, BasicCoordinate<num_dimensions, coordT>& v)
{
  char c = '\0';
  str >> std::ws >> c;
  if (!str || c != '{')
  {
    warning("reading a coordinate of dimension %d, expected opening {, found %c instead.\n"
              "Elements will be undefined", num_dimensions, c);
    return str;
  }
  for (int i=1; i<=num_dimensions; i++)
  { 
    c = '\0';
    str >> v[i];
    str >> std::ws >> c;
    if (i<num_dimensions && (!str || c!=','))
    {
      warning("reading a coordinate of dimension %d, expected comma, found %c instead.\n"
              "Remaining elements will be undefined", num_dimensions, c);
      return str;
    }
    if (i==num_dimensions && (!str || c!='}'))
    {
      warning("reading a coordinate of dimension %d, expected closing }, found %c instead.",num_dimensions, c);
      return str;
    }
  }
  return str;
  
}

END_NAMESPACE_STIR
