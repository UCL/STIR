/*!
  \file
  \ingroup buildblock

  \brief Functions useful for manipulating Interfile keywords

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_interfile_keyword_functions_H__ 
#define __stir_interfile_keyword_functions_H__ 
#include "stir/common.h"


#include <string>
#include <functional>

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
std::string 
standardise_interfile_keyword(const std::string& keyword);

//! A function object that compares Interfile keywords
/*! This is similar to std::less<std::string>, except that it applies 
    standardise_interfile_keyword() on its arguments before doing the
    comparison.

    Useful for  constructing a std::map for instance.
*/
struct interfile_less : public std::binary_function<std::string, std::string, bool>
{
  bool operator()(const std::string& a, const std::string& b) const
  {
    return 
      standardise_interfile_keyword(a) < 
      standardise_interfile_keyword(b);
  }
};

END_NAMESPACE_STIR

#endif
