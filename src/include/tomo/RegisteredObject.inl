//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Inline implementations for class RegisteredObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/
START_NAMESPACE_TOMO
 

template <typename Root>
RegisteredObject<Root>::RegisteredObject()
{}


#ifndef _MSC_VER
template <typename Root>
RegisteredObject<Root>::RegistryType& 
RegisteredObject<Root>::registry ()
{
  static RegistryType the_registry("None", 0);
  return the_registry;
}
#endif

template <typename Root>
Root*
RegisteredObject<Root>::read_registered_object(istream* in, const string& registered_name)
{
  RootFactory factory = registry().find_factory(registered_name);
  return factory==0 ? 0 : (*factory)(in);
}

template <typename Root>
void 
RegisteredObject<Root>::
list_registered_names(ostream& stream)
{
  registry().list_keys(stream);
}



END_NAMESPACE_TOMO
