//
// $Id$
//

#ifndef __stir_recon_buildblock_ProjMatrixElemsForOneDensel__
#define __stir_recon_buildblock_ProjMatrixElemsForOneDensel__

/*!

  \file
  \ingroup recon_buildblock
  
  \brief Declaration of class stir::ProjMatrixElemsForOneDensel
    
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



#include "stir/recon_buildblock/ProjMatrixElemsForOneDenselValue.h"
#include "stir/Densel.h"
#include <vector>
//#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::vector;
//using std::fstream;
using std::random_access_iterator_tag;
#endif



START_NAMESPACE_STIR

class RelatedDensels;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;



/*! \ingroup projection
\brief This stores the non-zero projection matrix elements
  for every 'voxel'.

  Most of the members of this class would work just as well
  for a (not yet existing) class ProjMatrixElemsForOneVoxel.
  This means that we should derived both from a common
  base class, templated in the type of element (TODO).

  It might be useful to template this class in terms of the
  element-type as well. That way, we could have 'compact' 
  elements, efficient elements, etc. However, doing this
  will probably only be useful if all ProjMatrixByDensel classes
  are then templated as well (TODO?).
*/

/* 
  it might be a bit faster to derive this (privately) from
  vector<value_type> as opposed to having a member of
  that type.
  TODO: check
*/
class ProjMatrixElemsForOneDensel 
{  
public:
  /*! \brief Recommended way to call the type of the elements, instead of 
  referring to the actual classname.

  Think about this name as 'the type of the value of a ProjMatrixElemsForOneDensel::iterator *'.

  This typedef is also required for 'standard' iterators.
  */
  typedef ProjMatrixElemsForOneDenselValue value_type;
private:
  //! shorthand to keep typedefs below concise
  typedef vector<value_type> Element_vector;

public:  
  //! typedefs for iterator support
  typedef Element_vector::iterator iterator;  
  typedef Element_vector::const_iterator const_iterator;  
  typedef Element_vector::size_type size_type;
  typedef Element_vector::difference_type difference_type;
  typedef random_access_iterator_tag iterator_category;

  typedef value_type& reference;
  typedef const value_type& const_reference;


  //! constructor
  ProjMatrixElemsForOneDensel(); 
  /*!
    \param Densel effectively calls set_densel(Densel)
    \param default_capacity effectively calls reserve(default_capacity)
  */
  explicit ProjMatrixElemsForOneDensel(const Densel& Densel, const int default_capacity = 300); 
  
  /* rely on compiler-generated versions 
  ProjMatrixElemsForOneDensel( const ProjMatrixElemsForOneDensel&);
  ProjMatrixElemsForOneDensel& operator=(const ProjMatrixElemsForOneDensel&) ;
  */

  //! check if each voxel occurs only once
  Succeeded check_state() const;
  
  //! get the Densel coordinates corresponding to this row
  inline Densel get_densel() const;
  //! and set the Densel coordinates
  inline void set_densel(const Densel&);

  //! functions for allowing iterator access
  inline iterator begin() ;
  inline const_iterator  begin() const;
  inline iterator end();
  inline const_iterator end() const;

  //! reset lor to 0 length
  void erase();
  //! add a new value_type object at the end
  /*! 
     \warning For future compatibility, it is required 
     (but not checked) that the elements are added such 
     that calling sort() after the push_back() would not change
     the order of the elements. Otherwise, schemes for
     'incremental' storing of coordinates would require too
     much overhead.
     */
  inline void push_back( const value_type&);    	
  //! reserve enough space for max_number elements (but don't fill them in)
  void reserve(size_type max_number);
  //! number of non-zero elements
  inline size_type size() const;	

  //! Multiplies all values with a constant
  ProjMatrixElemsForOneDensel& operator*=(const float d); 
  //! Divides all values with a constant
  ProjMatrixElemsForOneDensel& operator/=(const float d); 
  
  //! Sort the elements on coordinates of the voxels
  /*! Uses value_type::coordinates_less as ordering function.
  */
  void sort();

  //! merge 2nd lor into current object
  /*! This makes sure that in the result, no duplicate coordinates occur.
     \warning This currently modifies the argument \c lor.
     */
  // TODO make sure we can have a const argument
  void merge(ProjMatrixElemsForOneDensel &lor );

#if 0  
  void write(fstream&fst) const;     	
  void read(fstream&fst );
#endif
  

  //! Return sum of squares of all values
  /*! \warning This sums over all elements in the LOR, irrespective if they
      are inside the FOV or not
  */
  float square_sum() const;

  //******************** projection operations ********************//
#if 0
  //! back project a single Densel 
  void back_project(DiscretisedDensity<3,float>&,
                    const Densel&) const;

  //! forward project into a single Densel
  void forward_project(Densel&,
                      const DiscretisedDensity<3,float>&) const;
 //! back project related Densels
  void back_project(DiscretisedDensity<3,float>&,
                    const RelatedDensels&) const; 
  //! forward project related Densels
  void forward_project(RelatedDensels&,
                       const DiscretisedDensity<3,float>&) const;

#endif
  
private:
  vector<value_type> elements;    
  Densel densel;


  //! remove a single value_type
  inline iterator erase(iterator it);
};


END_NAMESPACE_STIR

#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.inl"

#endif
