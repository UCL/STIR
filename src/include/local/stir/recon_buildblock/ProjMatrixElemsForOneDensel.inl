//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief 

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
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
