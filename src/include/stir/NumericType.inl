//
// $Id$
//
/*!
  \file 
  \ingroup buildblock 
  \brief Implementation of inline methods of class NumericType.

  \author Kris Thielemans 
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR

NumericType::NumericType(Type t)
: id(t)
{}

bool NumericType::operator==(NumericType type) const
{ 
  return id == type.id; 
}

bool NumericType::operator!=(NumericType type) const
{ 
  return !(*this == type); 
}

END_NAMESPACE_STIR
