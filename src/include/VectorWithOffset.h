// VectorWithOffset.h 
// like vector.h, but with indices starting not from 0
// by KT, based on Tensor1D.h by DH
// TODO, add iterators a la STL

// $Id$ : $Date$

#ifndef VECTORWITHOFFSET_H
#define  VECTORWITHOFFSET_H

#include "pet_common.h"

template <class T>
class VectorWithOffset {

protected:

  Int length;	// length of matrix (in cells)
  Int start;	// vertical starting index

  T *num;	// array to hold elements indexed by start
  T *mem;	// pointer to start of memory for new/delete

  void Init() {		// Default member settings for all constructors
    length =0;	// i.e. an empty row of zero length,
    start = 0;	// no offsets
    num = mem = 0;				// and no data.
  };

  //KT 13/11 added this function, only non-empty when debugging
  //to be used before and after any modification of the object
  // KT 21/11 had to make this protected now, as it is called by
  // Tensorbase
  void check_state() const
  { assert(((length > 0) ||
            (length == 0 && start == 0 &&
             num == 0 && mem == 0)));
  }

public:  
	
  void Recycle() {	// Free memory and make object as if default-constructed
    check_state();
    if (length > 0){
      delete[] mem; 
      Init();
    }
  };
	
  VectorWithOffset() { 
    Init();
  };

  // Construct a VectorWithOffset of given length
  //KT TODO don't know how to write this constructor in terms of the more general one below
  VectorWithOffset(const Int hsz) {	
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
      if (length > 0) {
	mem = new T[length];
	num = mem - hfirst;
      } else Init(); 
    check_state();
  }

  ~VectorWithOffset() { Recycle(); };		// Destructor

  void set_offset(const Int hfirst) {
    check_state();
    //KT 13/11 only allowed when non-zero length
    if (length == 0) return;
    start = hfirst;
    num = mem - start;
  }

  //grow the length range of the tensor, new elements are set to NUMBER()
  void grow(Int hfirst, Int hlast) { 
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
    delete [] mem;
    mem = newmem;
    num = newnum;
    length = new_length;
    start = hfirst;
    check_state();
  }

  // Assignment operator
#ifdef TEMPLATE_ARG
  VectorWithOffset & operator= (const VectorWithOffset<T, NUMBER> &il) 	
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
      }			       			// care of by Tensor::operator=
    else	Recycle();
    check_state();
    return *this;
  }

  // Copy constructor
#ifdef TEMPLATE_ARG
  VectorWithOffset(const VectorWithOffset<T, NUMBER> &il)
#else
  VectorWithOffset(const VectorWithOffset &il) 
#endif
  {
    Init();
    *this = il;		// Uses assignment operator (above)
  };

  Int get_length() const { check_state(); return length; };	// return length of VectorWithOffset

  //KT prefer this name to get_offset, because I need a name for the maximum index
  Int get_min_index() const { check_state(); return start; }
  Int get_max_index() const { check_state(); return start + length - 1; }

  T& operator[] (Int i) {	// Allow array-style access, read/write
    check_state();
    assert((i>=start)&&(i<(length+start)));
    return num[i];
  };
	
  //KT 13/11 return a reference now, avoiding copying
  const T& operator[] (Int i) const {  // array access, read-only
    check_state();
    //KT 13/11 can't return T() now anymore (reference to temporary)
    assert((i>=start)&&(i<(length+start)));
    return num[i];
    // if ((i>=start)&&(i<(length+start))) return num[i];
    ////KT somewhat strange design choice
    //else { return T(); }
  };
		
  // comparison
#ifdef TEMPLATE_ARG
  bool operator== (const VectorWithOffset<T, NUMBER> &iv) const
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
  
};

#endif
