//
// $Id$
//
#ifndef __stir_Succeeded_H__
#define __stir_Succeeded_H__

/*!

  \file
  \ingroup buildblock
  \brief Declaration of class stir::Succeeded

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
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

START_NAMESPACE_STIR

/*! 
  \brief 
  a class containing an enumeration type that can be used by functions to signal 
  successful operation or not

  Example:
  \code
  Succeeded f() { do_something;  return Succeeded::yes; }
  void g() { if (f() == Succeeded::no) error("Error calling f"); }
  \endcode
*/
class Succeeded
{
public:
  enum value { yes, no };
  Succeeded(const value& v) : v(v) {}
  bool operator==(const Succeeded &v2) const { return v == v2.v; }
  bool operator!=(const Succeeded &v2) const { return v != v2.v; }
private:
  value v;
};

END_NAMESPACE_STIR

#endif
