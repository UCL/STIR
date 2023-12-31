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
    Copyright (C) 2023, University College London
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
  // the latter can also be written as
  void g2() { if (!f().succeeded()) error("Error calling f"); }
  \endcode
*/
class Succeeded
{
public:
  enum value { yes, no };
  Succeeded(const value& v = yes) : v(v) {}
  bool operator==(const Succeeded &v2) const { return v == v2.v; }
  bool operator!=(const Succeeded &v2) const { return v != v2.v; }
  //! convenience function returns if it is equal to Succeeded::yes
  bool succeeded() const { return this->v == yes; }
private:
  value v;
};

END_NAMESPACE_STIR

#endif
