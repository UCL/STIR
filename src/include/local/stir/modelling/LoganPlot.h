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

#ifndef __stir_modelling_LoganPlot_H__
#define __stir_modelling_LoganPlot_H__

//#include "local/stir/decay_correct.h"
#include "local/stir/modelling/ModelMatrix.h"
#include "local/stir/modelling/BloodFrameData.h"
#include "local/stir/modelling/PlasmaData.h"

START_NAMESPACE_STIR

#define num_param 2

class LoganPlot
{
  public:

  //! default constructor
  inline LoganPlot();

  //! Create model matrix from blood frame data
  inline ModelMatrix<num_param> get_model_matrix(const PlasmaData& plasma_data,const TimeFrameDefinitions& time_frame_definitions,const unsigned int starting_frame);

  //! default destructor
  inline ~LoganPlot();

 private:
  ModelMatrix<num_param> _model_matrix;
  bool _matrix_is_stored;
};


END_NAMESPACE_STIR

#include "local/stir/modelling/LoganPlot.inl"

#endif //__stir_modelling_LoganPlot_H__
