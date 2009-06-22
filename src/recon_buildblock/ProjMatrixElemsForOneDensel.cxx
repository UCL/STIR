//
// $Id$
//
/*!

  \file
  \ingroup projection
  \brief non-inline implementations for stir::ProjMatrixElemsForOneDensel
 
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
#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/SymmetryOperation.h"
#include "stir/recon_buildblock/DataSymmetriesForDensels.h"
#include "stir/recon_buildblock/RelatedDensels.h"

#include <fstream>

START_NAMESPACE_STIR


ProjMatrixElemsForOneDensel::
ProjMatrixElemsForOneDensel(const Densel& densel, const int default_capacity)
: densel(densel)
{
  elements.reserve(default_capacity); 
}

ProjMatrixElemsForOneDensel::
ProjMatrixElemsForOneDensel()
{
  elements.reserve(300); 
}


void 
ProjMatrixElemsForOneDensel::
reserve(size_type max_number)
{
  elements.reserve(max_number);
}


void ProjMatrixElemsForOneDensel::erase()
{
  elements.resize(0);
}

ProjMatrixElemsForOneDensel& ProjMatrixElemsForOneDensel::operator*=(const float d)
{
  // KT 21/02/2002 added check on 1
  if (d != 1.F)
  {
    iterator element_ptr = begin();
    while (element_ptr != end())
    {
      *element_ptr *= d;        
      ++element_ptr;
    }	 
  }
  return *this;
}

ProjMatrixElemsForOneDensel& ProjMatrixElemsForOneDensel::operator/=(const float d)
{
  assert( d != 0);
  // KT 21/02/2002 added check on 1
  if (d != 1.F)
  {
    iterator element_ptr = begin();
    while (element_ptr != end())
    { 
      *element_ptr /= d;
      ++element_ptr;
    }
  }
  return *this;
}


Succeeded ProjMatrixElemsForOneDensel::check_state() const
{
  Succeeded success = Succeeded::yes;

  if (size()==0)
    return success;

  ProjMatrixElemsForOneDensel lor = *this;
  lor.sort();
  
  for (ProjMatrixElemsForOneDensel::const_iterator lor_iter = lor.begin();
       lor_iter != lor.end()-1; 
       ++lor_iter)
  {
    if (value_type::coordinates_equal(*lor_iter, *(lor_iter+1)))
    {
      warning("ProjMatrixElemsForOneDensel: coordinates occur more than once %d,%d,%d,%d\n",
	      lor_iter->segment_num(), lor_iter->view_num(), 
	      lor_iter->axial_pos_num(), lor_iter->tangential_pos_num());
      success = Succeeded::no;
    }
  }
  return success;
}


void ProjMatrixElemsForOneDensel::sort()
{
  // need explicit std:: here to resolve possible name conflict
  // this might give you trouble if your compiler does not support namespaces
#if !defined(STIR_NO_NAMESPACES) || (__GNUC__ == 2 && __GNUC_MINOR__ <= 8)
  std::
#endif                                           
  sort(begin(), end(), value_type::coordinates_less);
}


float ProjMatrixElemsForOneDensel::square_sum() const
{
  float sq_sum=0;
  const_iterator element_ptr = begin();
  while (element_ptr != end())
  {
    sq_sum += square(element_ptr->get_bin_value());        
    ++element_ptr;
  }	 
  return sq_sum;
}

// TODO make sure we can have a const argument
// not calling lor2.erase() would probably speed it up anyway
void 
ProjMatrixElemsForOneDensel::
merge( ProjMatrixElemsForOneDensel &lor2 )
{
  assert(check_state() == Succeeded::yes);
  assert(lor2.check_state() == Succeeded::yes);

  iterator element_ptr = begin();
  iterator element_ptr2= lor2.begin();
  
  bool found=false;
  while ( element_ptr2 != lor2.end() )
  {   		
    //unsigned int key = make_key( element_ptr2->x, element_ptr2->y,element_ptr2->z);    		 
    iterator  dup_xyz = element_ptr; 	  
    while ( dup_xyz != end() )
    {
      //unsigned int dup_key = make_key( dup_xyz->x,dup_xyz->y,dup_xyz->z);			 
      
      //if ( dup_key == key ) 
      if (value_type::coordinates_equal(*element_ptr2, *dup_xyz))
      {
        //TEST/////////
        *dup_xyz += *element_ptr2;		    	  
        ///////////////
        element_ptr = dup_xyz+1;
        found = true;
        break; // assume no more duplicated point, only 2 lors 
      }
      else
        ++dup_xyz;
    }
    if( found )
    {	        
      element_ptr2  = lor2.erase(element_ptr2);
      found = false;
    }
    else{            
      ++element_ptr2;			
    }	
  }
  // append the rest
  element_ptr2  = lor2.begin();
  while ( element_ptr2 != lor2.end() )
  {   		  
    push_back( *element_ptr2);
    ++element_ptr2;
  }
  assert(check_state() == Succeeded::yes);
}


#if 0
// todo remove this 
void ProjMatrixElemsForOneDensel::clean_neg_z()
{	
iterator element_ptr = begin();
while (element_ptr != end())
{  
if ( element_ptr->z  < 0 || element_ptr->z > _maxplane ) {
element_ptr = elements.erase(element_ptr); 
cout << " !!!!!! clean !!!!!! " << element_ptr->z  << endl;
		}
        else 
	  	  ++element_ptr;		
	}	 
	
}
#endif

#if 0
void ProjMatrixElemsForOneDensel::write (fstream&fst) const  
{  
  //fst.write ( (char*)&_sgn, sizeof( unsigned int));
  int c= size();
  fst.write( (char*)&c , sizeof( int));  
  const_iterator element_ptr = begin();
  // todo add compression in this loop 
  while (element_ptr != end()) {           
    fst.write ( (char*)element_ptr, sizeof( value_type));
    ++element_ptr;
  } 
  //  fst.write ( (char*)&_minplane, sizeof( int)); 
  //  fst.write ( (char*)&_maxplane, sizeof( int));        
} 

//! Read probabilities from stream
void ProjMatrixElemsForOneDensel::read(   fstream&fst )
{  
 //fst.read ( (char*)&_sgn, sizeof( unsigned int) );
  
  int count;
  fst.read ( (char*)&count, sizeof( int));  	 
  // todo handel the compression 
  for ( int i=0; i < count; ++i) { 
    value_type elem;
    fst.read ( (char*)&elem, sizeof( value_type));
    push_back( elem);		
  }  
  //fst.read ( (char*)&_minplane, sizeof( int));
  //fst.read ( (char*)&_maxplane, sizeof( int));   
  // print();     
}
#endif

/////////////////// projection  operations ////////////////////////////////// 
#if 0
// TODO
void 
ProjMatrixElemsForOneDensel::
back_project(DiscretisedDensity<3,float>& density,   
             const Densel& single) const
{   
  {  
    const float data = single.get_densel_value() ;     
    const_iterator element_ptr = 
      begin();
    while (element_ptr != end())
    {
      const BasicCoordinate<3,int> coords = element_ptr->get_coords();
      if (coords[1] >= density.get_min_index() && coords[1] <= density.get_max_index())
        density[coords[1]][coords[2]][coords[3]] += element_ptr->get_value() * data;		
      element_ptr++;            
    }    
  }    
}


void 
ProjMatrixElemsForOneDensel::
forward_project(Densel& single,
                const DiscretisedDensity<3,float>& density) const
{
  {  
    
    const_iterator element_ptr = begin();
    
    while (element_ptr != end())
    {
      const BasicCoordinate<3,int> coords = element_ptr->get_coords();
      
      if (coords[1] >= density.get_min_index() && coords[1] <= density.get_max_index())
        single += density[coords[1]][coords[2]][coords[3]] * element_ptr->get_value();
      ++element_ptr;		
    }	      
  }   
}


void 
ProjMatrixElemsForOneDensel::
back_project(DiscretisedDensity<3,float>& density,
             const RelatedDensels& r_densels) const                              
{  
  const DataSymmetriesForDensels* symmetries = r_densels.get_symmetries_ptr(); 
  
  RelatedDensels::const_iterator r_densels_iterator =r_densels.begin();
  ProjMatrixElemsForOneDensel row_copy;
  while (r_densels_iterator != r_densels.end())
    
  {    
    row_copy = *this;
    
    Densel symmetric_densel = *r_densels_iterator;
    auto_ptr<SymmetryOperation> symm_ptr = 
      symmetries->find_symmetry_operation_to_basic_densel(symmetric_densel);
    symm_ptr->transform_proj_matrix_elems_for_one_densel(row_copy);
    row_copy.back_project(density,symmetric_densel);
  }  
}


void  
ProjMatrixElemsForOneDensel::
forward_project(RelatedDensels& r_densels,
                const DiscretisedDensity<3,float>& density) const
{
  const DataSymmetriesForDensels* symmetries = r_densels.get_symmetries_ptr(); 
  
  RelatedDensels::iterator r_densels_iterator =r_densels.begin();
  ProjMatrixElemsForOneDensel row_copy;
  
  while (r_densels_iterator != r_densels.end())
    
  {    
    row_copy = *this;
    
    auto_ptr<SymmetryOperation> symm_op_ptr = 
      symmetries->find_symmetry_operation_to_basic_densel(*r_densels_iterator);
    symm_op_ptr->transform_proj_matrix_elems_for_one_densel(row_copy);
    row_copy.forward_project(*r_densels_iterator,density);
  }  
  
}
#endif

END_NAMESPACE_STIR
