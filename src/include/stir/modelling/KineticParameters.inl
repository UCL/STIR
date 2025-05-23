//
//
/*
    Copyright (C) 2006 - 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::KineticParameters

  \author Charalampos Tsoumpas

*/

START_NAMESPACE_STIR

  //! default constructor
template <int num_param, typename elemT>
KineticParameters<num_param,elemT>::KineticParameters()
{ }
  //! constructor
//template <int num_param, typename elemT>
//KineticParameters<int num_param, typename elemT>::KineticParameters()
//{}

  //! default destructor
template <int num_param, typename elemT>
KineticParameters<num_param,elemT>::~KineticParameters()
{ }

  //! set the blood counts of the sample
template <int num_param, typename elemT> 
void KineticParameters<num_param,elemT>::
set_parameter_value( const elemT param_value, const int param_num)
{ KineticParameters::_kin_params[param_num]=param_value ; }

  //! get the blood counts of the sample
template <int num_param, typename elemT> 
elemT KineticParameters<num_param,elemT>::
get_parameter_value(const int param_num) const
{  return KineticParameters::_kin_params[param_num] ; }


END_NAMESPACE_STIR
