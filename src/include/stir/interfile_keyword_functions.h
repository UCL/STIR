//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Functions useful for manipulating Interfile keywords

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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


#include <string>
#include <functional>

#ifndef STIR_NO_NAMESPACE
using std::string;
using std::binary_function;
#endif

START_NAMESPACE_STIR

//! Put a (Interfile) keyword into a standard form
/*!
  This follows Interfile 3.3 conventions:
  <ul>
  <li> The characters \c space, \c tab, \c underscore, \c ! are all
       treated as white space.       
  <li> Starting and trailing white space is trimmed.
  <li> Repeated white space is replaced with a single space
  <li> All letters are made lowercase.
  </ul>  
*/
string 
standardise_interfile_keyword(const string& keyword);

//! A function object that compares Interfile keywords
/*! This is similar to std::less<string>, except that it applies 
    standardise_interfile_keyword() on its arguments before doing the
    comparison.

    Useful for  constructing a std::map for instance.
*/
struct interfile_less : public binary_function<string, string, bool>
{
  bool operator()(const string& a, const string& b) const
  {
    return 
      standardise_interfile_keyword(a) < 
      standardise_interfile_keyword(b);
  }
};

END_NAMESPACE_STIR
