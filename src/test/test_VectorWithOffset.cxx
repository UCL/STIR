// $Id$: $Date$
// Test program for VectorWithOffset.h
// Version 1.0 : KT

#include "VectorWithOffset.h"
#ifdef __MSL__
#include <algorithm.h>
#include <numeric.h>
#else
#include <algo.h>
#endif

#include <iostream.h>

main()
{
  cerr << "Test program for VectorWithOffset.h.\n"
       << "Everythings is fine if the program runs without any output." << endl;
  
  VectorWithOffset<int> v(-3, 40);

#ifdef DEFINE_ITERATORS
  iota(v.begin(), v.end(), -3);
  assert(v[4] == 4);

  {
    int *p=find(v.begin(), v.end(), 6);
    assert(p - v.begin() == 9);
    assert(v.get_index(p) == 6);
  }

  {
    VectorWithOffset<int>::reverse_iterator pr = v.rbegin();
    assert((*pr++) == 40);
    assert((*pr++) == 39);
  }

#if !(defined( __GNUG__)  && (__GNUC__ < 2 || (__GNUC__ == 2 && __GNUC_MINOR__ < 8)))
  // sadly doesn't work under g++ 2.7
  sort(v.rbegin(), v.rend());
  assert(v[-3] == 40);
  assert(v[0] == 37);
#endif
#endif // DEFINE_ITERATORS

}
