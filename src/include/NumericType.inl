//
// $Id$: $Date$
//

/* 
  Implementation of inline methods of class NumericType
  
  History:
  - first version Kris Thielemans

*/

NumericType::NumericType(Type t)
: id(t)
{}

bool NumericType::operator==(NumericType type) const
{ 
  return id == type.id; 
}
