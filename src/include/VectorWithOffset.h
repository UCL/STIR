// $Id$: $Date$

#ifndef __VECTORWITHOFFSET_H__
#define __VECTORWITHOFFSET_H__
// like vector.h, but with indices starting not from 0
// by KT, based on Tensor1D.h by DH

// KT 26/11 added iterator things

/*
   KT 27/03/98 use two levels of debugging. 
   For normal debugging #define _DEBUG. 
   For extensive debugging, use #define _DEBUG 2

   KT 26/10/98 new members get_data_ptr(), update_data(), release_data_ptr()
*/

#include "pet_common.h"
#include <iterator>

template <class T>
class VectorWithOffset
{
public:
  // like in vector.h and defalloc.h
  typedef T value_type;
  typedef T * pointer;
  typedef const T * const_pointer;
  typedef T * iterator;
  typedef const T * const_iterator;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
#ifdef DEFINE_ITERATORS
  //KT 22/01/98 updated to gcc 2.8.0
#ifdef __STL_CLASS_PARTIAL_SPECIALIZATION
  // the new ANSI C++ style reverse iterator
  typedef reverse_iterator<const_iterator>  const_reverse_iterator;
  typedef reverse_iterator<iterator> reverse_iterator;
#else
#ifdef __MSL__
  // Modena STL implementation
  typedef reverse_iterator<const_iterator, value_type, const_reference, 
  const_pointer, difference_type>  const_reverse_iterator;
  typedef reverse_iterator<iterator, value_type, reference, 
  pointer, difference_type> reverse_iterator;
#else
  // (old) HP STL implementation, used by gcc 2.7.2
  typedef reverse_iterator<const_iterator, value_type, const_reference, 
  difference_type>  const_reverse_iterator;
  typedef reverse_iterator<iterator, value_type, reference, difference_type>
  reverse_iterator;
#endif /* __MSL__ */
#endif /*  __STL_CLASS_PARTIAL_SPECIALIZATION */
#endif /* DEFINE_ITERATORS */
protected:

  Int length;	// length of matrix (in cells)
  Int start;	// vertical starting index

  T *num;	// array to hold elements indexed by start
  T *mem;	// pointer to start of memory for new/delete

  // Default member settings for all constructors
  void Init() 
    {		
      length =0;	// i.e. an empty row of zero length,
      start = 0;	// no offsets
      num = mem = 0;	// and no data.
    };

  //KT 13/11 added this function, only non-empty when debugging
  //to be used before and after any modification of the object
  // KT 21/11 had to make this protected now, as it is called by
  // Tensorbase
  void check_state() const
    {
      // KT 27/03/98 disable for normal debugging
#if _DEBUG>1
      assert(((length > 0) ||
	      (length == 0 && start == 0 &&
	       num == 0 && mem == 0)));
    
#endif
      // KT 26/10/98 new
      // check if data is being access via a pointer (see get_data_ptr())
      assert(pointer_access == false);
    }

public:  

  // Free memory and make object as if default-constructed	
  void Recycle() 
    {
      check_state();
      if (length > 0){
	delete[] mem; 
	Init();
      }
    };
	
  VectorWithOffset()
    { 
      // KT&AZ 26/10/98 new
      pointer_access = false;

      Init();
    };

  // Construct a VectorWithOffset of given length
  //KT TODO don't know how to write this constructor in terms of the more general one below
  VectorWithOffset(const Int hsz)
    {	
      // KT&AZ 26/10/98 new
      pointer_access = false;

      if ((hsz > 0)) {
	length = hsz;
	start = 0;
	num = mem = new T[hsz];
      } else Init();
      check_state();
    }			
    
  // Construct a VectorWithOffset of elements with offsets hfirst
  VectorWithOffset(const Int hfirst, const Int hlast)   
    : length(hlast - hfirst + 1),
      start(hfirst)
    { 
      // KT&AZ 26/10/98 new
      pointer_access = false;

      if (length > 0) {
	mem = new T[length];
	num = mem - hfirst;
      } else Init(); 
      check_state();
    }

  ~VectorWithOffset()
    { Recycle(); };		// Destructor

  void set_offset(const Int hfirst) 
    {
      check_state();
      //KT 13/11 only allowed when non-zero length
      if (length == 0) return;
      start = hfirst;
      num = mem - start;
    }

  // KT 02/03/98 made virtual (see Tensor3D.h why)
  //grow the length range of the tensor, new elements are set to NUMBER()
  virtual void grow(Int hfirst, Int hlast) { 
    check_state();
    const Int new_length = hlast - hfirst + 1;
    if (hfirst == start && new_length == length) {
      return;
    }

    //KT 13/11 grow arbitrary when it's zero length
    assert(length == 0 || (hfirst <= start && new_length >= length));
    T *newmem = new T[new_length];
    T *newnum = newmem - hfirst;
    Int i;
    //KTTODO the new members won't have the correct size (as the
    //default constructor is called. For the moment, we leave this to
    //Tensor3D and 4D.
    for (i=start ; i<start + length; i++)
      newnum[i] = num[i];
    // KT 18/12/97 added check on length. Probably not really necessary
    // as mem should be == 0 in that case, and delete[] 0 doesn't do anything
    // (I think). Anyway, we're on the safe side now...
    if (length != 0)
      delete [] mem;
    mem = newmem;
    num = newnum;
    length = new_length;
    start = hfirst;
    check_state();
  }

  // Assignment operator
#ifdef TEMPLATE_ARG
  VectorWithOffset & operator= (const VectorWithOffset<NUMBER> &il) 	
#else
    VectorWithOffset & operator= (const VectorWithOffset &il) 
#endif
    {
      check_state();
      if (this == &il) return *this;		// in case of x=x
      if (il.length > 0)
	{		
	  if (length != il.length)	// if new tensorbase has different
	    {				// length, reallocate memory
	      //KT TODO optimisation possible (skipping a superfluous Init() )
	      //if (length > 0) delete [] mem;
	      //in fact, the test on length can be skipped, because when
	      //length == 0, mem == 0, and delete [] 0 doesn't do anything
	      //???check
	      Recycle();
	      length = il.length;
	      mem = new T[length];
	    }
	  set_offset(il.get_min_index());
	  for(Int i=0; i<length; i++)     
	    mem[i] = il.mem[i];		// different widths are taken 
	}			      	// care of by Tensor::operator=
      else	
	Recycle();
      check_state();
      return *this;
    }

  // Copy constructor
#ifdef TEMPLATE_ARG
  VectorWithOffset(const VectorWithOffset<NUMBER> &il)
#else
    VectorWithOffset(const VectorWithOffset &il) 
#endif
    {
      // KT&AZ 29/10/98 new
      pointer_access = false;

      Init();
      *this = il;		// Uses assignment operator (above)
    };

  Int get_length() const { check_state(); return length; };	// return length of VectorWithOffset

  //KT prefer this name to get_offset, because I need a name for the maximum index
  Int get_min_index() const { check_state(); return start; }
  Int get_max_index() const { check_state(); return start + length - 1; }

  T& operator[] (Int i) {	// Allow array-style access, read/write
    check_state();
    // KT 26/10/98 split into 2 assertions
    assert(i>=start);
    assert(i<(length+start));

    return num[i];
  };
	
  //KT 13/11 return a reference now, avoiding copying
  const T& operator[] (Int i) const {  // array access, read-only
    check_state();
    // KT 26/10/98 split into 2 assertions
    assert(i>=start);
    assert(i<(length+start));

    //KT 13/11 can't return T() now anymore (reference to temporary)
    return num[i];
    // if ((i>=start)&&(i<(length+start))) return num[i];
    ////KT somewhat strange design choice
    //else { return T(); }
  };
		
  // comparison
#ifdef TEMPLATE_ARG
  bool operator== (const VectorWithOffset<NUMBER> &iv) const
#else
    bool operator== (const VectorWithOffset &iv) const
#endif
    {
      check_state();
      if (length != iv.length || start != iv.start) return false;
      for (Int i=0; i<length; i++)
	if (mem[i] != iv.mem[i]) return false;
      return true; }

  // Fill elements with value n
  void fill(const T &n) 
    {
      check_state();
      for(Int i=0; i<length; i++)
	mem[i] = n;
      check_state();
    };
  
#ifdef DEFINE_ITERATORS
  // KT 26/11 added iterator things here
  iterator begin() { return mem; }
  const_iterator begin() const { return mem; }
  iterator end() { return mem+length; }
  const_iterator end() const { return mem+length; }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const { 
    return const_reverse_iterator(end()); 
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { 
    return const_reverse_iterator(begin()); 
  }

  // returns the index such that &v[v.get_index(iter)]  == iter
  Int get_index(iterator iter)
    { return iter - num; }
#endif // DEFINE_ITERATORS

  // KT&AZ 29/10/98 made public
  
  // KT&AZ 26/10/98 new

  /* Members for access to the data via a T*.
     To prevent invalidating the safety checks (and making 
     reimplementation more difficult), NO manipulation with
     the vector is allowed between the pairs
     get_data_ptr() and update_data() 
     get_data_ptr() and release_data_ptr()
     get_const_data_ptr() and release_data_ptr().
     */

  
  bool pointer_access;

  T* get_data_ptr()
    {
      // KT&AZ 29/10/98 added assert
      assert(!pointer_access);

      pointer_access = true;
      return mem;

      // if implementation changes, this would need to keep track 
      // if which pointer it returns.
    };

  void update_data()
    {
      // KT&AZ 29/10/98 added assert
      assert(pointer_access);

      pointer_access = false;
      // do nothing now
    };

  const T * get_const_data_ptr()
    {
      // KT&AZ 29/10/98 added assert
      assert(!pointer_access);

      pointer_access = true;
      return mem;

      // if implementation changes, this would need to keep track 
      // if which pointer it returns.
    };

  void release_data_ptr()
    {
      // KT&AZ 29/10/98 added assert
      assert(pointer_access);

      pointer_access = false;
    }
};

#ifdef DEFINE_ITERATORS
#if defined (__GNUG__) && (__GNUC__ < 2 || (__GNUC__ == 2 && __GNUC_MINOR__ < 8))
// Work arounds from vector.h for G++ 2.7
// sort() etc. use value_type(), which needs the lines below.
// Note that value_type() of a reverse_iterator does not work (also not
// in vector.h
template <class T>

struct VectorWithOffset_iterator 
{
  VectorWithOffset<T>::iterator it;
  VectorWithOffset_iterator(VectorWithOffset<T>::iterator i)
    : it(i) {}
  operator VectorWithOffset<T>::iterator()
    {
      return it;
    }
};

template <class T>
inline T* value_type(const VectorWithOffset_iterator<T>&) 
{
  return (T*)(0);
}


template <class T>
struct VectorWithOffset_const_iterator
{
  VectorWithOffset<T>::const_iterator it;
  VectorWithOffset_const_iterator(VectorWithOffset<T>::const_iterator i)
    : it(i) {}
  operator VectorWithOffset<T>::const_iterator() {
    return it;
  }
};

// KT: I included these to make things work with rend() and so on. 
// For some reason it's not enough. Here is an illustration:
/* 
 VectorWithOffset<int> v(1,10);
 value_type(v.begin());
 value_type( VectorWithOffset_reverse_iterator<int>(v.rbegin()));
 // next line doesn't compile. Missing an automatic conversion.
 value_type(v.rbegin());
 */
template <class T>
struct VectorWithOffset_reverse_iterator 
{
  VectorWithOffset<T>::reverse_iterator it;
  VectorWithOffset_reverse_iterator(VectorWithOffset<T>::reverse_iterator i)
    : it(i) {}
  operator VectorWithOffset<T>::reverse_iterator() {
    return it;
  }
};

template <class T>
inline T* value_type(const VectorWithOffset_reverse_iterator<T>&) 
{
  return (T*)(0);
}


template <class T>
struct VectorWithOffset_const_reverse_iterator 
{
  VectorWithOffset<T>::const_reverse_iterator it;
  VectorWithOffset_const_reverse_iterator(VectorWithOffset<T>::const_reverse_iterator i) 
    : it(i) {}
  operator VectorWithOffset<T>::const_reverse_iterator() {
    return it;
  }
};
#endif // check on GNU 2.7
#endif // DEFINE_ITERATORS

#endif
