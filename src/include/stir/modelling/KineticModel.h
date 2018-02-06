//
//
/*
    Copyright (C) 2006 - 2009, Hammersmith Imanet Ltd
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

  \brief Definition of class stir::KineticModel

  \author Charalampos Tsoumpas
 
*/

#ifndef __stir_modelling_KineticModel_H__
#define __stir_modelling_KineticModel_H__

#include "stir/RegisteredObject.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*! 
  \brief base class for all kinetic models
  \ingroup modelling

  At present very basic. It just provides the parsing mechanism.
*/
class KineticModel: public RegisteredObject<KineticModel> 
{ 

public:
  static const char * const registered_name ; 
   //! default constructor
  KineticModel();

  //! default destructor
  virtual ~KineticModel();

  //  virtual float get_compartmental_activity_at_time(const int param_num, const int sample_num) const;
  //  virtual float get_total_activity_at_time(const int sample_num) const;

  virtual Succeeded set_up() = 0;

  // protected:
  //  void initialise_keymap();

};

END_NAMESPACE_STIR

#endif //__stir_modelling_KineticModel_H__
