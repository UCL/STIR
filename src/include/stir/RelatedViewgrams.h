//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class RelatedViewgrams

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#ifndef __RelatedViewgrams_h__
#define __RelatedViewgrams_h__

#include "Viewgram.h"
#include "DataSymmetriesForViewSegmentNumbers.h"
#include <vector>

#include <iterator>

#ifndef TOMO_NO_NAMESPACES
using std::size_t;
using std::ptrdiff_t;
using std::random_access_iterator_tag;
using std::vector;
#endif

START_NAMESPACE_TOMO

class PMessage;

// forward declarations for 'friend'
class ProjData;
class ProjDataInfo;



/*!
  \brief A class for storing viewgrams which are related by symmetry  
  \ingroup buildblock
*/
template <typename elemT>
class RelatedViewgrams
{

public:
  //! typedefs for iterator support


 typedef random_access_iterator_tag iterator_category;
 typedef Viewgram<elemT> value_type;
 typedef value_type& reference;
 typedef const value_type& const_reference;
 typedef ptrdiff_t difference_type;
 typedef size_t size_type;

#ifndef TOMO_NO_NAMESPACES
  typedef std::vector<Viewgram<elemT> >::iterator iterator;
  typedef std::vector<Viewgram<elemT> >::const_iterator const_iterator;
#else
  typedef vector<Viewgram<elemT> >::iterator iterator;
  typedef vector<Viewgram<elemT> >::const_iterator const_iterator;
#endif



  // --- constructors ---

  //! default constructor (sets everything empty)
  inline RelatedViewgrams();    

  // implicit copy constructor (just element-by-element copy)
  // RelatedViewgrams(const RelatedViewgrams&);

  //! serialisation ctor
  RelatedViewgrams(PMessage& msg);



  // --- const members returning info ---

  //! get 'basic' view_num
  /*! see DataSymmetriesForViewSegmentNumbers for definition of 'basic' */
  inline int get_basic_view_num() const;
  //! get 'basic' segment_num
  /*! see DataSymmetriesForViewSegmentNumbers for definition of 'basic' */
  inline int get_basic_segment_num() const;
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
  //! Get a pointer to the symmetries used in constructing this object
  inline const DataSymmetriesForViewSegmentNumbers * get_symmetries_ptr() const;
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

 
private:
  
  friend class ProjData;
  friend class ProjDataInfo;

  // members
  vector<Viewgram<elemT> > viewgrams;
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_used;

  //! a function which is called internally to see if the object is valid
  /*! it does nothing when NDEBUG is defined */
  inline void check_state() const;

  //! actual implementation of the above function
  void debug_check_state() const;

  //! a private constructor which simply sets the members
  inline RelatedViewgrams(const vector<Viewgram<elemT> >& viewgrams,
                   const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used);

};

END_NAMESPACE_TOMO

#include "RelatedViewgrams.inl"


#endif // __RelatedViewgrams_h__

