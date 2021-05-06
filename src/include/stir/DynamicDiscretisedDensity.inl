//
//
/*!
  \file
  \ingroup densitydata
  \brief Inline implementations of class stir::DynamicDiscretisedDensity
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
*/
/*
    Copyright (C) 2005- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


START_NAMESPACE_STIR

DynamicDiscretisedDensity::full_iterator DynamicDiscretisedDensity::begin_all()
{
  return full_iterator(this->_densities.begin(), this->_densities.end());
}

DynamicDiscretisedDensity::const_full_iterator DynamicDiscretisedDensity::begin_all() const
{
  return const_full_iterator(this->_densities.begin(), this->_densities.end());
}

DynamicDiscretisedDensity::const_full_iterator DynamicDiscretisedDensity::begin_all_const() const
{
  return const_full_iterator(this->_densities.begin(), this->_densities.end());
}

DynamicDiscretisedDensity::full_iterator DynamicDiscretisedDensity::end_all()
{
  return full_iterator(this->_densities.end(), this->_densities.end());
}

DynamicDiscretisedDensity::const_full_iterator DynamicDiscretisedDensity::end_all() const
{
  return const_full_iterator(this->_densities.end(), this->_densities.end());
}

DynamicDiscretisedDensity::const_full_iterator DynamicDiscretisedDensity::end_all_const() const
{
  return const_full_iterator(this->_densities.end(), this->_densities.end());
}

END_NAMESPACE_STIR
