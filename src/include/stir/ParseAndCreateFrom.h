/*
    Copyright (C) 2019, University College London
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
  \ingroup densitydata 
  
  \brief  Definition of the stir::ParseAndCreateFrom class
    
  \author Kris Thielemans
*/

#ifndef __stir_ParseAndCreateFrom_H__
#define __stir_ParseAndCreateFrom_H__

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ParseDiscretisedDensityParameters.h"
START_NAMESPACE_STIR

class KeyParser;

//! template for adding keywords to a parser and creating an object
/*!
  \ingroup buildblock

  The idea is that a reconstructor needs to be able to create an image based
  on some parameters and the current input data. However, this needs to be flexible
  for different types, as for instance, many different reconstructors produce
  a DiscretisedDensity object but from different input data, or vice versa.

  We do this using specialisations of this class. That way, 
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData etc can be templated
  without having to know what the actual \c OutputT is.

  This of course only works if a specialisation is created for the
  \c OutputT and \InputT of interest.

  A specialisation needs to define all four member functions as the reconstruction
  code will otherwise break.

  The default implementations don't do anything (aside from create()).

  Check ParseAndCreateFrom<DiscretisedDensity<3, elemT>, ExamDataT> for an example
  which might make sense.
*/
template <class OutputT, class InputT, class ParserT = KeyParser>
class ParseAndCreateFrom
{
 public:
 //! set default values for any parameters
 void set_defaults() {}
 //! add any relevant parameters to a parser
 void add_to_keymap(ParserT&) {}
  //! should call error() if something is wrong
 void check_values() const {};

 //! create a new object
 /*! 
   This can take any parsed parameters into account.

   The default just calls \c new.

   \todo Currently we're assuming this returns a bare pointer (to a new object).
   This is due to limitations in the reconstruction classes. It will need to change
   to a \c std::unique pointer.
 */
 OutputT* create(const InputT&) const
 { return new OutputT(); }
};


//! parse keywords for creating a VoxelsOnCartesianGrid from ProjData etc
/*!
  \ingroup densitydata 

  This specialisation adds keywords like size etc to the parser, and calls
  the VoxelsOnCartesianGrid constructor with ExamInfo and ProjDataInfo arguments
  to obtain an image that is suitable to store the reconstruction.

  Assumes that \c ExamDataT has \c get_exam_info_sptr() and \c get_proj_data_info_ptr()
  members.

  \todo Currently only supports VoxelsOnCartesianGrid parameters (we could introduce
  another keyword to differentiate between types).
*/
template <class elemT, class ExamDataT>
  class ParseAndCreateFrom<DiscretisedDensity<3, elemT>, ExamDataT>
  : public ParseDiscretisedDensityParameters
{
 public:
  typedef DiscretisedDensity<3, elemT> output_type;
  inline
  output_type*
    create(const ExamDataT&) const;
};

END_NAMESPACE_STIR

#include "stir/ParseAndCreateFrom.inl"
#endif
