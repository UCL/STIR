//
//
#ifndef __stir_Succeeded_H__
#define __stir_Succeeded_H__

/*!

  \file
  \ingroup buildblock
  \brief Declaration of class stir::Succeeded

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
