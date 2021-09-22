//
//
/*!
  \file 
  \ingroup buildblock 
  \brief Implementation of inline methods of class stir::NumericType.

  \author Kris Thielemans 
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

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
