//
//
/*!

  \file
  \ingroup projection

  \brief Inline implementations for class stir::ProjMatrixElemsForOneDensel

  \author Kris Thielemans


*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
