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
*/
/*!
  \file
  \ingroup modelling

  \brief This is a preliminary version of the abstract class stir::KineticModel<num_param> which should be as generic as possible to accommodate all the Kinetic Models. 

  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/

#ifndef __stir_modelling_KineticModel_H__
#define __stir_modelling_KineticModel_H__

#include "stir/RegisteredObject.h"
#include "stir/RegisteredParsingObject.h"
//#include "local/stir/modelling/ModelMatrix.h"

START_NAMESPACE_STIR

//template <int num_param>
class KineticModel: public RegisteredObject<KineticModel> 
{ 

public:
  static const char * const registered_name ; 
   //! default constructor
  KineticModel();

  //! default destructor
  ~KineticModel();

  //  virtual float get_compartmental_activity_at_time(const int param_num, const int sample_num) const;
  //  virtual float get_total_activity_at_time(const int sample_num) const;

  // protected:
  //  void initialise_keymap();

  // private:
  //  ModelMatrix<2> _model_matrix; // But it may not be as simple as in the case of linear models.
};

END_NAMESPACE_STIR

#endif //__stir_modelling_KineticModel_H__
