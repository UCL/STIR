/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2012, Hammersmith Imanet Ltd
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
  \ingroup projdata
  \brief Declaration of class stir::RelatedViewgrams

  \author Kris Thielemans
  \author PARAPET project
*/

#ifndef __RelatedViewgrams_h__
#define __RelatedViewgrams_h__

#include "stir/Viewgram.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include <vector>

#include <iterator>

START_NAMESPACE_STIR

// forward declarations for 'friend'
class ProjData;
class ProjDataInfo;



/*!
  \brief A class for storing viewgrams which are related by symmetry  
  \ingroup projdata
*/
template <typename elemT>
class RelatedViewgrams
{
private:
#ifdef SWIG
public:  
#endif
  typedef RelatedViewgrams<elemT> self_type;

public:
  //! \name typedefs for iterator support
  //@{
  typedef std::random_access_iterator_tag iterator_category;
  typedef Viewgram<elemT> value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef std::ptrdiff_t difference_type;
  typedef std::size_t size_type;

  typedef typename std::vector<Viewgram<elemT> >::iterator iterator;
  typedef typename std::vector<Viewgram<elemT> >::const_iterator const_iterator;
  //@}


  // --- constructors ---

  //! default constructor (sets everything empty)
  inline RelatedViewgrams();    

  // implicit copy constructor (just element-by-element copy)
  // RelatedViewgrams(const RelatedViewgrams&);
 
  //! a private constructor which simply sets the members
  /*! \todo Currently public for the STIR_MPI version */
  inline RelatedViewgrams(const std::vector<Viewgram<elemT> >& viewgrams,
                   const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used);


  // --- const members returning info ---

  //! get 'basic' view_num
  /*! see DataSymmetriesForViewSegmentNumbers for definition of 'basic' */
  inline int get_basic_view_num() const;
  //! get 'basic' segment_num
  /*! see DataSymmetriesForViewSegmentNumbers for definition of 'basic' */
  inline int get_basic_segment_num() const;
  //! get 'basic' timing_pos_num
  /*! see DataSymmetriesForViewSegmentNumbers for definition of 'basic' */
  inline int get_basic_timing_pos_num() const;
  //! get 'basic' view_segment_num
  /*! see DataSymmetriesForViewSegmentNumbers for definition of 'basic' */
  inline ViewSegmentNumbers get_basic_view_segment_num() const;

  //! returns the number of viewgrams in this object
  inline int get_num_viewgrams() const;
  inline int get_num_axial_poss() const;
  inline int get_num_tangential_poss() const;
  inline int get_min_axial_pos_num() const;
  inline int get_max_axial_pos_num() const;
  inline int get_min_tangential_pos_num() const;
  inline int get_max_tangential_pos_num() const;

  //! Get a pointer to the ProjDataInfo of this object
  inline const ProjDataInfo * get_proj_data_info_ptr() const;
  //! Get shared pointer to proj data info
  /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
    shared pointer will be affected. */
  inline shared_ptr<ProjDataInfo>
    get_proj_data_info_sptr() const;
  //! Get a pointer to the symmetries used in constructing this object
  inline const DataSymmetriesForViewSegmentNumbers * get_symmetries_ptr() const;
  //! Get a shared pointer to the symmetries used in constructing this object
  /*! \warning It is dangerous to modify the shared symmetries object */
  inline shared_ptr<DataSymmetriesForViewSegmentNumbers> get_symmetries_sptr() const;
  // -- members which modify the structure ---

  //! Grow each viewgram
  void grow(const IndexRange<2>& range);

  //TODOvoid zoom(const float zoom, const float Xoffp, const float Yoffp,
  //          const int size, const float itophi);



  // -- basic iterator support --

  //! use to initialise an iterator to the first element of the vector
   inline iterator begin();
   //! iterator 'past' the last element of the vector
   inline iterator end();
    //! use to initialise an iterator to the first element of the (const) vector
   inline const_iterator begin() const;
   //! iterator 'past' the last element of the (const) vector
   inline const_iterator end() const;



   // numeric operators

   //! Multiplication of all data elements with a constant
   RelatedViewgrams& operator*= (const elemT);
   //! Division of all data elements by a constant
   RelatedViewgrams& operator/= (const elemT);
   //! Addition of all data elements by a constant
   RelatedViewgrams& operator+= (const elemT);
   //! Subtraction of all data elements by a constant
   RelatedViewgrams& operator-= (const elemT);


   //! Element-wise multiplication with another RelatedViewgram object
   RelatedViewgrams& operator*= (const RelatedViewgrams<elemT>&);
   //! Element-wise division by another RelatedViewgram object
   RelatedViewgrams& operator/= (const RelatedViewgrams<elemT>&);
   //! Element-wise addition by another RelatedViewgram object
   RelatedViewgrams& operator+= (const RelatedViewgrams<elemT>&);
   //! Element-wise subtraction by another RelatedViewgram object
   RelatedViewgrams& operator-= (const RelatedViewgrams<elemT>&);

   // numeric functions

   //! Find the maximum of all data elements
   elemT find_max() const;
   //! Find the maximum of all data elements
   elemT find_min() const;
   //! Set all data elements to n
   void fill(const elemT &n);

   // other

   //! Return a new object with ProjDataInfo etc., but all data elements set to 0
   RelatedViewgrams get_empty_copy() const;

  //! \name Equality
  //@{
  //! Checks if the 2 objects have the proj_data_info, segment_num etc.
  /*! If they do \c not have the same characteristics, the string \a explanation
      explains why.
  */
  bool
    has_same_characteristics(self_type const&,
			     std::string& explanation) const;

  //! Checks if the 2 objects have the proj_data_info, segment_num etc.
  /*! Use this version if you do not need to know why they do not match.
   */
  bool
    has_same_characteristics(self_type const&) const;

  //! check equality (data has to be identical)
  /*! Uses has_same_characteristics() and Array::operator==.
      \warning This function uses \c ==, which might not be what you 
      need to check when \c elemT has data with float or double numbers.
  */
  bool operator ==(const self_type&) const; 
  
  //! negation of operator==
  bool operator !=(const self_type&) const; 
  //@}
 
private:
  
  friend class ProjData;
  friend class ProjDataInfo;

  // members
  std::vector<Viewgram<elemT> > viewgrams;
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_used;

  //! a function which is called internally to see if the object is valid
  /*! it does nothing when NDEBUG is defined */
  inline void check_state() const;

  //! actual implementation of the above function
  void debug_check_state() const;

};

END_NAMESPACE_STIR

#include "stir/RelatedViewgrams.inl"


#endif // __RelatedViewgrams_h__

