//
//
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
/*!

  \file
  \ingroup projection
  \brief non-inline implementations for stir::ProjMatrixElemsForOneBin
 
  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
  
*/
/* History

   KT 15/05/2002 
   rewrote merge(). it worked only on sorted arrays before, while the end-result wasn't sorted
*/
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/SymmetryOperation.h"
#include "stir/recon_buildblock/DataSymmetriesForBins.h"

#include "stir/recon_buildblock/RelatedBins.h"
#include <algorithm>
//#include <iostream>
//#include "stir/stream.h"

#ifndef STIR_NO_NAMESPACES
using std::copy;
#endif

START_NAMESPACE_STIR


ProjMatrixElemsForOneBin::
ProjMatrixElemsForOneBin(const Bin& bin, const int default_capacity)
: bin(bin)
{
  elements.reserve(default_capacity); 
}


void 
ProjMatrixElemsForOneBin::
reserve(size_type max_number)
{
  elements.reserve(max_number);
}


void ProjMatrixElemsForOneBin::erase()
{
  elements.resize(0);
}

ProjMatrixElemsForOneBin::size_type 
ProjMatrixElemsForOneBin::
capacity() const 
{
  return elements.capacity();
}

ProjMatrixElemsForOneBin& ProjMatrixElemsForOneBin::operator*=(const float d)
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

ProjMatrixElemsForOneBin& ProjMatrixElemsForOneBin::operator/=(const float d)
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


Succeeded ProjMatrixElemsForOneBin::check_state() const
{
  Succeeded success = Succeeded::yes;

  if (size()==0)
    return success;

  ProjMatrixElemsForOneBin lor = *this;
  lor.sort();
  
  for (ProjMatrixElemsForOneBin::const_iterator lor_iter = lor.begin();
       lor_iter != lor.end()-1; 
       ++lor_iter)
  {
    if (value_type::coordinates_equal(*lor_iter, *(lor_iter+1)))
    {
      warning("ProjMatrixElemsForOneBin: coordinates occur more than once %d,%d,%d for bin s=%d, tofbin=%d, v=%d, a=%d, t=%d\n",
	      lor_iter->coord1(), lor_iter->coord2(), lor_iter->coord3(),
	      bin.segment_num(), bin.timing_pos_num(), bin.view_num(),
	      bin.axial_pos_num(), bin.tangential_pos_num());
#if 0
      const_iterator iter = begin();
      while (iter!= end())
	{
	  std::cerr << iter->get_coords() 
		    << ':' << iter->get_value()
		    << '\n';
	  ++iter;
	}
#endif
      success = Succeeded::no;
    }
  }
  return success;
}


void ProjMatrixElemsForOneBin::sort()
{
  // need explicit std:: here to resolve possible name conflict
  // this might give you trouble if your compiler does not support namespaces
#if !defined(STIR_NO_NAMESPACES) || (__GNUC__ == 2 && __GNUC_MINOR__ <= 8)
  std::
#endif                                           
  sort(begin(), end(), value_type::coordinates_less);
}


float ProjMatrixElemsForOneBin::square_sum() const
{
  float sq_sum=0;
  const_iterator element_ptr = begin();
  while (element_ptr != end())
  {
    sq_sum += square(element_ptr->get_value());        
    ++element_ptr;
  }	 
  return sq_sum;
}

// TODO make sure we can have a const argument
void 
ProjMatrixElemsForOneBin::
merge( ProjMatrixElemsForOneBin &lor2 )
{
  assert(check_state() == Succeeded::yes);
  assert(lor2.check_state() == Succeeded::yes);

  if (lor2.size()==0)
    return;

#ifndef NDEBUG
  // we will check at the end of the total probability is conserved, so we compute it first
  float old_sum = 0;
  for (const_iterator iter= begin(); iter != end(); ++iter)
    old_sum+= iter->get_value();
  for (const_iterator iter= lor2.begin(); iter != lor2.end(); ++iter)
    old_sum+= iter->get_value();
#endif
  // KT 15/05/2002 insert sort first, as both implementations assume this
  sort();
  lor2.sort();

  iterator element_ptr = begin();
  iterator element_ptr2= lor2.begin();
  // implementation note: cannot use const_iterator in the line above
  // as some compilers (including gcc 3.* and VC 7.0) would not compile
  // the elements.insert() line below, as the types of the iterators no longer 
  // match exactly.
#if 1
  // KT 15/05/2002 much faster implementation than before, and its output is sorted
#ifndef NDEBUG
  //unsigned int insertions = 0;
#endif
  while ( element_ptr2 != lor2.end() )
  {
    if (element_ptr == end())
    {
      elements.reserve((lor2.end() - element_ptr2) + size());
      elements.insert(end(), element_ptr2, lor2.end()); 
      break;
    }
    if (value_type::coordinates_equal(*element_ptr2, *element_ptr))
    {
      *element_ptr++ += *element_ptr2++;
    }
    else if (value_type::coordinates_less(*element_ptr2, *element_ptr))
    {
      element_ptr = elements.insert(element_ptr, *element_ptr2);
      assert(*element_ptr == *element_ptr2);
      ++element_ptr; ++element_ptr2;
#ifndef NDEBUG
      //++insertions;
#endif
    } 
    else
    {
      ++element_ptr;
    }
  }


#else
  // this old version assumed the input arrays were sorted, but its output wasn't
  // it could be rewritten to avoid lor2.erase() such that this method could be const
  // this would probably speed it up anyway
  bool found=false;
  while ( element_ptr2 != lor2.end() )
  {
    // here it is assumed that the input is sorted
    // this could be possibly be dropped by initialising dup_xyz = begin()
    // performance would be bad however
    iterator  dup_xyz = element_ptr;
    while ( dup_xyz != end() )
    {
      if (value_type::coordinates_equal(*element_ptr2, *dup_xyz))
      {
        *dup_xyz += *element_ptr2;		    	  
        element_ptr = dup_xyz+1;
        found = true;
        break; // assume no more duplicated point
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
  // append the rest (but not sorted)
  element_ptr2  = lor2.begin();
  while ( element_ptr2 != lor2.end() )
  {   		  
    push_back( *element_ptr2);
    ++element_ptr2;
  }
  // KT 15/05/2002 added sort for compatiblity with other version
  sort();
#endif
#ifndef NDEBUG
  // check that the values sum to the same as before merging
  float new_sum = 0;
  for (const_iterator iter= begin(); iter != end(); ++iter)
    new_sum+= iter->get_value();
  assert(fabs(new_sum-old_sum)<old_sum*10E-4);
  // check that it's sorted
  for (const_iterator iter=begin(); iter+1!=end(); ++iter)
    assert(value_type::coordinates_less(*iter, *(iter+1)));
  //std::cerr << insertions << " insertions, size " << size() << std::endl;
#endif
  assert(check_state() == Succeeded::yes);
}


#if 0
// todo remove this 
void ProjMatrixElemsForOneBin::clean_neg_z()
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
void ProjMatrixElemsForOneBin::write (fstream&fst) const  
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
void ProjMatrixElemsForOneBin::read(   fstream&fst )
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
void 
ProjMatrixElemsForOneBin::
back_project(DiscretisedDensity<3,float>& density,   
             const Bin& single) const
{   
  {  
    const float data = single.get_bin_value() ;     
    // KT 21/02/2002 added check on 0
    if (data == 0)
      return;
    
    BasicCoordinate<3,int> coords;
    const_iterator element_ptr = 
      begin();
    while (element_ptr != end())
    {
      coords = element_ptr->get_coords();
      if (coords[1] >= density.get_min_index() && coords[1] <= density.get_max_index())
        density[coords[1]][coords[2]][coords[3]] += element_ptr->get_value() * data;		
      element_ptr++;            
    }    
  }    
}


void 
ProjMatrixElemsForOneBin::
forward_project(Bin& single,
                const DiscretisedDensity<3,float>& density) const
{
  {  
    
    BasicCoordinate<3,int> coords;
    const_iterator element_ptr = begin();
    
    while (element_ptr != end())
    {
      coords = element_ptr->get_coords();
      
      if (coords[1] >= density.get_min_index() && coords[1] <= density.get_max_index())
        single += density[coords[1]][coords[2]][coords[3]] * element_ptr->get_value();
      ++element_ptr;		
    }	      
  }   
}


void 
ProjMatrixElemsForOneBin::
back_project(DiscretisedDensity<3,float>& density,
             const RelatedBins& r_bins) const                              
{  
  const DataSymmetriesForBins* symmetries = r_bins.get_symmetries_ptr(); 
  
  RelatedBins::const_iterator r_bins_iterator =r_bins.begin();
  ProjMatrixElemsForOneBin row_copy;
  while (r_bins_iterator != r_bins.end())
    
  {    
    row_copy = *this;
    
    Bin symmetric_bin = *r_bins_iterator;
    // KT 21/02/2002 added check on 0
    if (symmetric_bin.get_bin_value() == 0)
      return;
    unique_ptr<SymmetryOperation> symm_ptr = 
      symmetries->find_symmetry_operation_from_basic_bin(symmetric_bin);
    symm_ptr->transform_proj_matrix_elems_for_one_bin(row_copy);
    row_copy.back_project(density,symmetric_bin);
  }  
}


void  
ProjMatrixElemsForOneBin::
forward_project(RelatedBins& r_bins,
                const DiscretisedDensity<3,float>& density) const
{
  const DataSymmetriesForBins* symmetries = r_bins.get_symmetries_ptr(); 
  
  RelatedBins::iterator r_bins_iterator =r_bins.begin();
  ProjMatrixElemsForOneBin row_copy;
  
  while (r_bins_iterator != r_bins.end())
    
  {    
    row_copy = *this;
    
    unique_ptr<SymmetryOperation> symm_op_ptr = 
      symmetries->find_symmetry_operation_from_basic_bin(*r_bins_iterator);
    symm_op_ptr->transform_proj_matrix_elems_for_one_bin(row_copy);
    row_copy.forward_project(*r_bins_iterator,density);
  }  
  
}


bool 
ProjMatrixElemsForOneBin::
operator==(const ProjMatrixElemsForOneBin& lor) const
{
  // first determine a tolerance for the floating point comparisons
  // do this by finding the maxumum value in the first lor
  float max_value = 0;

  {
    // extra brackets here to avoid VC 7.1 complaining about this_iter
    // being defined here in for() and a few lines further on
    // (it really shouldn't complain)
    for (const_iterator this_iter= begin(); this_iter!= end(); ++this_iter)
      {
	const float current_value=this_iter->get_value();
	if (current_value> max_value)
	  max_value = current_value;
      }
  }
  const float tolerance = max_value*.002F;

  const_iterator this_iter= begin(); 
  const_iterator lor_iter = lor.begin();
  while (this_iter!= end() && lor_iter!=lor.end())
    {
      // skip small elements
      if (this_iter->get_value()<tolerance)
	{ ++this_iter; continue; }
      if (lor_iter->get_value()<tolerance)
	{ ++lor_iter; continue; }
      // compare coordinates and value
      if (this_iter->get_coords() != lor_iter->get_coords() ||
	  fabs(this_iter->get_value() - lor_iter->get_value()) > tolerance)
	return false;
      ++this_iter;
      ++lor_iter;
    }
  for (; this_iter!= end(); ++this_iter)
    {
      if (this_iter->get_value()> tolerance)
	return false;
    }
  for (; lor_iter!= lor.end(); ++lor_iter)
    {
      if (lor_iter->get_value()> tolerance)
	return false;
    }
  return true;
  
}
bool 
ProjMatrixElemsForOneBin::
operator!=(const ProjMatrixElemsForOneBin& lor) const
{ return !(*this==lor); }

END_NAMESPACE_STIR
