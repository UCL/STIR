//
//

#ifndef __stir_recon_buildblock_ProjMatrixElemsForOneBin__
#define __stir_recon_buildblock_ProjMatrixElemsForOneBin__

/*!

  \file
  \ingroup projection
  
  \brief Declaration of class stir::ProjMatrixElemsForOneBin
    
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



#include "stir/recon_buildblock/ProjMatrixElemsForOneBinValue.h"
#include "stir/Bin.h"
#include <vector>

START_NAMESPACE_STIR

class Succeeded;
class RelatedBins;
template <int num_dimensions, typename elemT> class DiscretisedDensity;




/*!
\brief This stores the non-zero projection matrix elements
  for every 'densel' that contributes to a given bin.

  In the usual terminology, this class implements a Line (or Tube)
  of Response (LOR, or TOR).

  \todo Most of the members of this class would work just as well
  for a (not yet existing) class ProjMatrixElemsForOneDensel.
  This means that we should derived both from a common
  base class, templated in the type of element.

  \todo
  It might be useful to template this class in terms of the
  element-type as well. That way, we could have 'compact' 
  elements, efficient elements, etc. However, doing this
  will probably only be useful if all ProjMatrixByBin classes
  are then templated as well, which would be a pain.
*/

/* 
  it might be a bit faster to derive this (privately) from
  std::vector<value_type> as opposed to having a member of
  that type.
  TODO: check
*/
class ProjMatrixElemsForOneBin 
{  
public:
  /*! \brief Recommended way to call the type of the elements, instead of 
  referring to the actual classname.

  Think about this name as 'the type of the value of a ProjMatrixElemsForOneBin::iterator *'.

  This typedef is also required for 'standard' iterators.
  */
  typedef ProjMatrixElemsForOneBinValue value_type;
private:
  //! shorthand to keep typedefs below concise
  typedef std::vector<value_type> Element_vector;

public:  
  //! typedefs for iterator support
  typedef Element_vector::iterator iterator;  
  typedef Element_vector::const_iterator const_iterator;  
  typedef Element_vector::size_type size_type;
  typedef Element_vector::difference_type difference_type;
  typedef std::random_access_iterator_tag iterator_category;

  typedef value_type& reference;
  typedef const value_type& const_reference;


  //! constructor
  /*!
    \param bin effectively calls set_bin(bin)
    \param default_capacity effectively calls reserve(default_capacity)
  */
  explicit ProjMatrixElemsForOneBin(const Bin& bin= Bin(), const int default_capacity = 0); 
  
  /* rely on compiler-generated versions 
  ProjMatrixElemsForOneBin( const ProjMatrixElemsForOneBin&);
  ProjMatrixElemsForOneBin& operator=(const ProjMatrixElemsForOneBin&) ;
  */

  //! check if each voxel occurs only once
  Succeeded check_state() const;
  
  //! get the bin coordinates corresponding to this row
  inline Bin get_bin() const;
  //! and set the bin coordinates
  inline void set_bin(const Bin&);

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
  //! number of allocated elements
  size_type capacity() const;
  //! Multiplies all values with a constant
  ProjMatrixElemsForOneBin& operator*=(const float d); 
  //! Divides all values with a constant
  ProjMatrixElemsForOneBin& operator/=(const float d); 
  
  //! Sort the elements on coordinates of the voxels
  /*! Uses value_type::coordinates_less as ordering function.
  */
  void sort();

  //! merge 2nd lor into current object
  /*! This makes sure that in the result, no duplicate coordinates occur.
     \warning This currently modifies the argument \c lor.
     */
  // TODO make sure we can have a const argument
  void merge(ProjMatrixElemsForOneBin &lor );

  //! Compare 2 lors to see if they are equal
  /*! \warning Compares element by element. Does not sort first or so.
      \warning Compares float values, so uses a tolerance. This tolerance
      is currently set to a fraction of the maximum value in the first lor.
      \warning this is a fairly CPU intensive operation.
  */
  bool operator==(const ProjMatrixElemsForOneBin&) const;
  //! Compare 2 lors 
  bool operator!=(const ProjMatrixElemsForOneBin&) const;


#if 0  
  void write(std::fstream&fst) const;     	
  void read(std::fstream&fst );
#endif
  

  //! Return sum of squares of all values
  /*! \warning This sums over all elements in the LOR, irrespective if they
      are inside the FOV or not
  */
  float square_sum() const;

  //******************** projection operations ********************//

  //! back project a single bin 
  void back_project(DiscretisedDensity<3,float>&,
                    const Bin&) const;

  //! forward project into a single bin
  void forward_project(Bin&,
                      const DiscretisedDensity<3,float>&) const;
 //! back project related bins
  void back_project(DiscretisedDensity<3,float>&,
                    const RelatedBins&) const; 
  //! forward project related bins
  void forward_project(RelatedBins&,
                       const DiscretisedDensity<3,float>&) const;

  
private:
  std::vector<value_type> elements;    
  Bin bin;


  //! remove a single value_type
  inline iterator erase(iterator it);
};


END_NAMESPACE_STIR

#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.inl"

#endif
