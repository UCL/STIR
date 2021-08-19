//
//
/*!
  \file
  \ingroup buildblock
  \brief Inline implementations for class stir::RegisteredParsingObject

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR

template <typename Derived, typename Base, typename Parent>
std::string 
RegisteredParsingObject<Derived,Base,Parent>:: get_registered_name() const
  { return Derived::registered_name; }

template <typename Derived, typename Base, typename Parent>
Base*
RegisteredParsingObject<Derived,Base,Parent>::read_from_stream(std::istream* in)
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
std::string
RegisteredParsingObject<Derived,Base,Parent>::parameter_info() 
{
  return ParsingObject::parameter_info();
}

END_NAMESPACE_STIR
