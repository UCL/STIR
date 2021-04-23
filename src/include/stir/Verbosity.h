//
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class stir::Verbosity

  Derived from http://www.yolinux.com/TUTORIALS/C++Singleton.html
  and http://sourcemaking.com/design_patterns/singleton/cpp/1

  \author Matthias Ehrhardt

*/
/*
    Copyright (C) 2014, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_Verbosity_H__
#define __stir_Verbosity_H__
#include "stir/common.h"

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief This class enables the user to control the on-screen output
*/
class Verbosity
{
 public:
  static int get();
  static void set(int level);

 private:
  Verbosity(); // Private so that it can not be called

  int _verbosity_level;
  static Verbosity* _instance;
};

END_NAMESPACE_STIR
#endif
