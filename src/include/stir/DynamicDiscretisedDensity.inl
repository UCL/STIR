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
