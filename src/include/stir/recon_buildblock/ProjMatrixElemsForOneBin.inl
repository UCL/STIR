//
//
/*!

  \file
  \ingroup projection

  \brief Inline implementations for class stir::ProjMatrixelemesForOneBin

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

Bin 
ProjMatrixElemsForOneBin::
get_bin() const
{
  return bin;
}

Bin*
ProjMatrixElemsForOneBin::
get_bin_ptr()
{
  return &bin;
}

void
ProjMatrixElemsForOneBin::
set_bin(const Bin& new_bin)
{
  bin = new_bin;
}

void ProjMatrixElemsForOneBin::push_back( const ProjMatrixElemsForOneBin::value_type& el)
{  
  elements.push_back(el); 
}


ProjMatrixElemsForOneBin::size_type 
ProjMatrixElemsForOneBin::
size() const 
{
  return elements.size();
}

ProjMatrixElemsForOneBin::iterator  
ProjMatrixElemsForOneBin::begin()   
  {  return elements.begin(); }

ProjMatrixElemsForOneBin::const_iterator  
ProjMatrixElemsForOneBin::
begin() const  
  {  return elements.begin(); };

ProjMatrixElemsForOneBin::iterator 
ProjMatrixElemsForOneBin::
end()
  {  return elements.end();	};

ProjMatrixElemsForOneBin::const_iterator 
ProjMatrixElemsForOneBin::
end() const
  {  return elements.end();	};

ProjMatrixElemsForOneBin::iterator 
ProjMatrixElemsForOneBin::
erase(iterator it){
    return elements.erase(it);
  }

#if 0
unsigned int  ProjMatrixElemsForOneBin::make_key(int X,int Y,int Z) 	
{  	   
  // make x and y  positive
  // min_x and min_y is supposed to be > -300  
  X+= 300;    // assert :  mix_x  >= -300, max_x < 2^10 -300 
  Y+= 300;    // ....	   
  assert ( X >=0 && X < 0x03FF  -300);
  assert ( Y >=0 && Y < 0x03FF  -300);
  assert ( Z >= 0 &&  Z < 0x03FF );         //  Z >=0   & < 2^10	   
  return ( ((unsigned int)X<<20) | ((unsigned int)Y<<10) | (unsigned int)Z );
}
#endif
END_NAMESPACE_STIR
