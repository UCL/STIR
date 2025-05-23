//
//
/*
    Copyright (C) 2006 - 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::KineticModel

  \author Charalampos Tsoumpas

*/

#ifndef __stir_modelling_OneParamModel_H__
#define __stir_modelling_OneParamModel_H__

#include "stir_experimental/modelling/ModelMatrix.h"

START_NAMESPACE_STIR

class OneParamModel
{
  public:

  //! default constructor
  inline OneParamModel();

  //! constructor
  inline OneParamModel(const int starting_frame, const int last_frame);

  //! Create a unit model matrix for a single frame and single parameter 
  inline ModelMatrix<1> get_unit_matrix(const int starting_frame, const int last_frame);

  //! default destructor
  inline ~OneParamModel();

 private:
  ModelMatrix<1> _unit_matrix;
  int _starting_frame;
  int _last_frame;
  bool _matrix_is_stored;
};

END_NAMESPACE_STIR

#include "stir_experimental/modelling/OneParamModel.inl"

#endif //__stir_modelling_OneParamModel_H__
