//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Inline implementations for class RegisteredParsingObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

#include <fstream>

#ifndef TOMO_NO_NAMESPACE
using std::ifstream;
#endif

START_NAMESPACE_TOMO

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
    der_ptr->parse(*in);
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

END_NAMESPACE_TOMO
