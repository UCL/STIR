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
#include "tomo/common.h"


#include <string>
#include <functional>

#ifndef TOMO_NO_NAMESPACE
using std::string;
using std::binary_function;
#endif

START_NAMESPACE_TOMO

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

END_NAMESPACE_TOMO
