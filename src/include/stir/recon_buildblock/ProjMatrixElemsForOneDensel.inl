//
// $Id$
//
/*!

  \file
  \ingroup projection

  \brief Inline implementations for class stir::ProjMatrixElemsForOneDensel

  \author Kris Thielemans

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

Densel 
ProjMatrixElemsForOneDensel::
get_densel() const
{
  return densel;
}

void
ProjMatrixElemsForOneDensel::
set_densel(const Densel& new_densel)
{
  densel = new_densel;
}

void ProjMatrixElemsForOneDensel::push_back( const ProjMatrixElemsForOneDensel::value_type& el)
{  
  elements.push_back(el); 
}


ProjMatrixElemsForOneDensel::size_type 
ProjMatrixElemsForOneDensel::
size() const 
{
  return elements.size();
}

ProjMatrixElemsForOneDensel::iterator  
ProjMatrixElemsForOneDensel::begin()   
  {  return elements.begin(); }

ProjMatrixElemsForOneDensel::const_iterator  
ProjMatrixElemsForOneDensel::
begin() const  
  {  return elements.begin(); };

ProjMatrixElemsForOneDensel::iterator 
ProjMatrixElemsForOneDensel::
end()
  {  return elements.end();	};

ProjMatrixElemsForOneDensel::const_iterator 
ProjMatrixElemsForOneDensel::
end() const
  {  return elements.end();	};

ProjMatrixElemsForOneDensel::iterator 
ProjMatrixElemsForOneDensel::
erase(iterator it){
    return elements.erase(it);
  }


END_NAMESPACE_STIR
