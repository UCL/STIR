//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Inline implementations for class RegisteredParsingObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include <fstream>

#ifndef STIR_NO_NAMESPACE
using std::ifstream;
#endif

START_NAMESPACE_STIR

template <typename Derived, typename Base, typename Parent>
string 
RegisteredParsingObject<Derived,Base,Parent>:: get_registered_name() const
  { return Derived::registered_name; }

template <typename Derived, typename Base, typename Parent>
Base*
RegisteredParsingObject<Derived,Base,Parent>::read_from_stream(istream* in)
{
  Derived * der_ptr = new Derived;
  if (in != NULL)
  {
    if(der_ptr->parse(*in)==false)
    {
      //parsing failed, return 0 pointer
      delete der_ptr;
      return 0;
    }
  }
  else
    der_ptr->ask_parameters();
  return der_ptr;
}

template <typename Derived, typename Base, typename Parent>
string
RegisteredParsingObject<Derived,Base,Parent>::parameter_info() 
{
  return ParsingObject::parameter_info();
}

END_NAMESPACE_STIR
