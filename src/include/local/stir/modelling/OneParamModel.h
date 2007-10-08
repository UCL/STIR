//
// $Id$
//
/*
    Copyright (C) 2006 - $Date$, Hammersmith Imanet Ltd
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

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::KineticModel

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

#ifndef __stir_modelling_OneParamModel_H__
#define __stir_modelling_OneParamModel_H__

#include "local/stir/modelling/ModelMatrix.h"

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

#include "local/stir/modelling/OneParamModel.inl"

#endif //__stir_modelling_OneParamModel_H__
