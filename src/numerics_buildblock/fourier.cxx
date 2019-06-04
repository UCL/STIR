/*!
  \file 
  \ingroup DFT
  \brief Functions for computing discrete fourier transforms

  \author Kris Thielemans
*/
/*
    Copyright (C) 2003 - 2005-01-17, Hammersmith Imanet Ltd

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
#include "stir/numerics/fourier.h"
#include "stir/round.h"
#include "stir/modulo.h"
#include "stir/array_index_functions.h"
START_NAMESPACE_STIR


template <typename T>
static void bitreversal(VectorWithOffset<T>& data)
{
  const int n=data.get_length();
  int   j=1;
  for (int i=0;i<n;++i) {
    if (j/2 > i) {
      std::swap(data[j/2],data[i]);
    }
    int m=n;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
}

/* We cache factors exp(i*_PI/pow(2,k)). They will be computed during the first
   call of the Fourier functions, and then stored in static arrays.
*/
// exparray[k][i] = exp(i*_PI/pow(2,k))
typedef VectorWithOffset<VectorWithOffset<std::complex<float> > > exparray_t;
static   exparray_t exparray;

static void init_exparray(const int k, const int pow2k)
{
  if (exparray.get_max_index() >= k && exparray[k].size()>0)
    return;

  if (exparray.get_max_index() <k)
    exparray.grow(0, k);
  exparray[k].grow(0,pow2k-1);
  for (int i=0; i< pow2k; ++i)
    exparray[k][i]= std::exp(std::complex<float>(0, static_cast<float>((i*_PI)/pow2k)));
}

// expminarray[k][i] = exp(-i*_PI/pow(2,k))
// obviously just the complex conjugate of exparray
static   exparray_t expminarray;

static void init_expminarray(const int k, const int pow2k)
{
  if (expminarray.get_max_index() >= k && expminarray[k].size()>0)
    return;

  if (expminarray.get_max_index() <k)
    expminarray.grow(0, k);
  expminarray[k].grow(0,pow2k-1);
  for (int i=0; i< pow2k; ++i)
    expminarray[k][i]= std::exp(std::complex<float>(0, static_cast<float>(-(i*_PI)/pow2k)));
}


/* First we define 1D fourier transforms of vectors with almost arbitrary
   element types.
   This is almost a straightforward 1D FFT implementation. The only tricky bit
   is to make sure that all operations are written in a way that is defined
   (and efficient) in the case that the element type is a vector again.
*/

template <typename T>
void fourier_1d(T& c, const int sign)
{
  if (c.size()==0) return;
  assert(c.get_min_index()==0);
  assert(sign==1 || sign ==-1);
  bitreversal(c);
  // find 'nn' which is such that length==2^nn
  const int nn=round(log(static_cast<double>(c.size()))/log(2.));
  if (c.get_length()!= round(pow(2.,nn)))
    error ("fourier_1d called with array length %d which is not 2^%d\n", c.size(), nn);

  int k=0;
  int pow2k = 1; // will be updated to be round(pow(2,k))
  const int pow2nn=c.get_length(); // ==round(pow(2,nn)); 
  for (; k<nn; ++k, pow2k*=2)
  {
    if (sign==1)
      init_exparray(k,pow2k);
    else
      init_expminarray(k,pow2k);
    const exparray_t& cur_exparray =
      sign==1? exparray : expminarray;      
    for (int j=0; j< pow2nn;j+= pow2k*2) 
      for (int i=0; i< pow2k; ++i)
      {
        typename T::reference c1= c[i + j];
        typename T::reference c2= c[i + j + pow2k];
        
        typename T::value_type const t1 = c1; 
	/* here is what we have to do:
            typename T::value_type const t2 = 
              c2*cur_exparray[k][i];
            c1 = t1+t2; c2 = t1-t2;
	 however, this would create an unnecessary copy of t2, which is 
	 potentially large.
	 So, we rewrite it without t2, and using operations that also 
	 work when using multi-dim arrays.
	 Note that for multi-dimensional arrays, this code involves 4 
	 loops over the same data.
	 Using expression templates would speed this up.
	*/
        c2 *= cur_exparray[k][i];
        c1 += c2;
        c2 *= -1;
        c2 += t1;
      }
  }
}

namespace detail {

/* A class that does the recursion for multi-dimensional arrays.

   This is done with a class because partial template specialisation is
   more powerful than function overloading.
*/
template <typename elemT>
struct fourier_auxiliary
{
  static void 
  do_fourier(VectorWithOffset<elemT >& c, const int sign)
  {
    fourier_1d(c, sign);
    const typename VectorWithOffset<elemT>::iterator iter_end = c.end();
    for (typename VectorWithOffset<elemT>::iterator iter = c.begin();
	 iter != iter_end;
	 ++iter)
      fourier(*iter, sign);
  }
};

// specialisation for the one-dimensional case
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template <typename elemT>
struct fourier_auxiliary<std::complex<elemT> >
{
  static void 
  do_fourier(VectorWithOffset<std::complex<elemT> >& c, const int sign)
  {
    fourier_1d(c, sign);
  }
};


#else  //no partial template specialisation

// we just list float and double explicitly
struct fourier_auxiliary<std::complex<float> >
{
  static void 
  do_fourier(VectorWithOffset<std::complex<float> >& c, const int sign)
  {
    fourier_1d(c, sign);
  }
};

struct fourier_auxiliary<std::complex<double> >
{
  static void 
  do_fourier(VectorWithOffset<std::complex<double> >& c, const int sign)
  {
    fourier_1d(c, sign);
  }
};
#endif

} // end of namespace detail

// now the fourier function is easy to define in terms of the class above
template <typename T>
void 
fourier(T& c, const int sign)
{
#if !defined(_MSC_VER) || _MSC_VER>1200
  detail::fourier_auxiliary<typename T::value_type>::do_fourier(c,sign);
#else
  detail::fourier_auxiliary<T::value_type>::do_fourier(c,sign);
#endif
}


/******************************************************************
 DFT of real data
*****************************************************************/



template <typename T>
Array<1,std::complex<T> >
fourier_1d_for_real_data(const Array<1,T>& v, const int sign)
//Array<1,std::complex<typename T::value_type> >
//fourier_1d_for_real_data(const T& v, const int sign)
{
  //typedef std::complex<typename T::value_type> complex_t;
  typedef std::complex<T> complex_t;
  if (v.size()==0) return Array<1,complex_t>();
  assert(v.get_min_index()==0);
  assert(sign==1 || sign ==-1);
  if (v.size()%2!=0)
    error("fourier_1d_of_real can only handle arrays of even length.\n");

  Array<1,complex_t> c;
  const unsigned int n = static_cast<unsigned int>(v.size()/2);
  // we reserve a range of 0,n here, such that 
  // resize(n) later doesn't reallocate and copy
  c.reserve(n+1);
  c.resize(n);
  // fill in complex numbers.
  // note: we need to divide by 2 in the final result. To save
  // some time, we do that already here.
  for (int i=0; i<c.get_length(); ++i)
    c[i] =  complex_t(v[2*i]/2, v[2*i+1]/2);

  fourier_1d(c, sign);

  //cout << "C: " << c;
  c.resize(n+1);
  for (unsigned int i=1; i<=n/2; ++i)
    {
      const complex_t t1 = 
	(c[i]+std::conj(c[n-i]));
      // TODO could get exp() from static exparray
      // the nice thing about this code that it works even when the length is not a power of 2
      // (but of course, the call to fourier_1d would currently abort in that case)
      const complex_t t2 = 			   
	std::exp(complex_t(0, static_cast<T>(sign*(i*_PI)/n-_PI/2)))*
	(c[i]-std::conj(c[n-i]));

      c[i] = (t1 + t2);
      c[n-i] = std::conj(t1-t2);
    }
  {
    const complex_t c0_copy = c[0];
    c[0]=(c0_copy.real() + c0_copy.imag())*2;
    c[n]=(c0_copy.real() - c0_copy.imag())*2;
  }
  return c;
}


template <typename T>
Array<1,T>
inverse_fourier_1d_for_real_data_corrupting_input(Array<1,std::complex<T> >& c, const int sign)
{
  typedef std::complex<T> complex_t;
  if (c.size()==0) return Array<1,T>();
  assert(c.get_min_index()==0);
  assert(sign==1 || sign ==-1);
  const int n = c.get_length()-1;
  if (n%2!=0)
    error("inverse_fourier_1d_of_real_data can only handle arrays of even length.\n");

  /* Problematic asserts to check that the imaginary part of c[0] and c[n] is 0
     Trouble is that it could be only approximately 0 (e.g. when calling 
     inverse_fourier_real_data on multi-dimensional arrays).
     The version below tries to circumvent this problem by comparing with the
     norm of c. That fails however when c is zero (up to numerical precision).
     We can only know this by looking at the higher dimensional array, but
     we don't have that one to our disposal in this function.
     So, I disabled the asserts.
  */
  //assert(fabs(c[0].imag())<=.001*norm(c.begin_all(),c.end_all())/sqrt(n+1.)); // note divide by n+1 to avoid division by 0
  //assert(fabs(c[n].imag())<=.001*norm(c.begin_all(),c.end_all())/sqrt(n+1.));
  for (int i=1; i<=n/2; ++i)
    {
      const complex_t t1 = (c[i]+std::conj(c[n-i]));
      // TODO could get exp() from static exparray
      const complex_t t2 = 			   
	std::exp(complex_t(0, static_cast<T>(-sign*(i*_PI)/n+_PI/2)))*
	(c[i]-std::conj(c[n-i]));

      c[i] = (t1 + t2);
      c[n-i] = std::conj(t1-t2);
    }
  {
    c[0]=complex_t((c[0].real() + c[n].real()),
		   (c[0].real() - c[n].real())
		   );
  }

  // now get rid of c[n] 
  c.resize(n);
  //cout << "\nC: " << c/4;
  inverse_fourier(c, sign);
  // extract real numbers.
  Array<1,T> v(2*n);
  for (int i=0; i<n; ++i)
    {
      v[2*i]= c[i].real()/2;
      v[2*i+1]= c[i].imag()/2;
    }
  return v;
}

template <typename T>
Array<1,T>
inverse_fourier_1d_for_real_data(const Array<1,std::complex<T> >& c, const int sign)
{
  Array<1,std::complex<T> > tmp(c);
  return inverse_fourier_1d_for_real_data_corrupting_input(tmp, sign);
}


// multi-dimensional case

namespace detail {
/* A class that does the recursion for multi-dimensional arrays.

   This is done with a class because partial template specialisation is
   more powerful than function overloading.
*/
template <int num_dimensions, typename elemT>
struct fourier_for_real_data_auxiliary
{
  static Array<num_dimensions,std::complex<elemT> >
  do_fourier_for_real_data(const Array<num_dimensions,elemT >& c, const int sign)
  {
    // complicated business to get index range which is as follows:
    // outer_dimension = outer_dimension of c
    // all other dimensions are as small as possible (to avoid reallocations)
    BasicCoordinate<num_dimensions, int> min_index, max_index;
    for (int d=2; d<=num_dimensions; ++d)
      min_index[d] = max_index[d] = 0;
    min_index[1] = c.get_min_index();
    max_index[1] = c.get_max_index();
    Array<num_dimensions, std::complex<elemT> > array(IndexRange<num_dimensions>(min_index, max_index));
    for (int i=c.get_min_index(); i<=c.get_max_index(); ++i)
      array[i] = fourier_for_real_data(c[i], sign);
    fourier_1d(array, sign);
    return array;
  }
  static Array<num_dimensions,elemT>
  do_inverse_fourier_for_real_data_corrupting_input(Array<num_dimensions,std::complex<elemT> >& c, const int sign)
  {
    inverse_fourier_1d(c, sign);
    // complicated business to get index range which is as follows:
    // outer_dimension = outer_dimension of c
    // all other dimensions are as small as possible (to avoid reallocations)
    BasicCoordinate<num_dimensions, int> min_index, max_index;
    for (int d=2; d<=num_dimensions; ++d)
      min_index[d] = max_index[d] = 0;
    min_index[1] = c.get_min_index();
    max_index[1] = c.get_max_index();
    Array<num_dimensions, elemT> array(IndexRange<num_dimensions>(min_index, max_index));

    for (int i=c.get_min_index(); i<=c.get_max_index(); ++i)
      array[i] = inverse_fourier_for_real_data_corrupting_input(c[i], sign);
    return array;
  }
};

// specialisation for the one-dimensional case
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template <typename elemT>
struct fourier_for_real_data_auxiliary<1,elemT>
{
  static Array<1,std::complex<elemT> >
  do_fourier_for_real_data(const Array<1,elemT>& c, const int sign)
  {
    return
      fourier_1d_for_real_data(c, sign);
  }
  static Array<1,elemT>
  do_inverse_fourier_for_real_data_corrupting_input(Array<1,std::complex<elemT> >& c, const int sign)
  {
    return
      inverse_fourier_1d_for_real_data_corrupting_input(c, sign);
  }
};


#else  //no partial template specialisation

// we just list float explicitly

struct fourier_for_real_data_auxiliary<1,float>
{
  static Array<1,std::complex<float> >
    do_fourier_for_real_data(const Array<1,float>& c, const int sign)
  {
    return
      fourier_1d_for_real_data(c, sign);
  }
  static Array<1,float>
    do_inverse_fourier_for_real_data_corrupting_input(Array<1,std::complex<float> >& c, const int sign)
  {
    return
      inverse_fourier_1d_for_real_data_corrupting_input(c, sign);
  }
};

#if 0 
/* Disabled double for now. 
 If you want to use double, you will probably have
 to make sure that Array<1,std::complex<double> > is instantiated.
 At time of writing, you would do this at the end of Array.h
 */
struct fourier_for_real_data_auxiliary<1,double>
{
  static Array<1,std::complex<double> >
    do_fourier_for_real_data(const Array<1,double>& c, const int sign)
  {
    return
      fourier_1d_for_real_data(c, sign);
  }
  static Array<1,double>
    do_inverse_fourier_for_real_data_corrupting_input(Array<1,std::complex<double> >& c, const int sign)
  {
    return
      inverse_fourier_1d_for_real_data_corrupting_input(c, sign);
  }
};
#endif // end of double

#endif // end of BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

} // end of namespace detail


// now the fourier_for_real_data function is easy to define in terms of the class above
template <int num_dimensions, typename T>
Array<num_dimensions,std::complex<T> >
fourier_for_real_data(const Array<num_dimensions,T>& c, const int sign)
{
  return
    detail::fourier_for_real_data_auxiliary<num_dimensions,T>::
    do_fourier_for_real_data(c,sign);
}


template <int num_dimensions, typename T>
Array<num_dimensions,T >
inverse_fourier_for_real_data_corrupting_input(Array<num_dimensions,std::complex<T> >& c, const int sign)
{
  return
  detail::fourier_for_real_data_auxiliary<num_dimensions,T>::
    do_inverse_fourier_for_real_data_corrupting_input(c,sign);
}

template <int num_dimensions, typename T>
Array<num_dimensions,T>
inverse_fourier_for_real_data(const Array<num_dimensions,std::complex<T> >& c, const int sign)
{
  Array<num_dimensions,std::complex<T> > tmp(c);
  return inverse_fourier_for_real_data_corrupting_input(tmp, sign);
}

template <int num_dimensions, typename T>
Array<num_dimensions, std::complex<T> > 
pos_frequencies_to_all(const Array<num_dimensions, std::complex<T> >& c)
{
  assert(c.is_regular());
  BasicCoordinate<num_dimensions, int> min_index, max_index;
  c.get_regular_range(min_index, max_index);
  // check min_indices are 0
  assert(min_index == (min_index*0));  
  max_index[num_dimensions]=max_index[num_dimensions]*2-1;
  Array<num_dimensions, std::complex<T> > result(IndexRange<num_dimensions>(min_index, max_index));
  
  BasicCoordinate<num_dimensions, int> index = min_index;
  const BasicCoordinate<num_dimensions, int> sizes = max_index+1;
  do
    {
      result[index] = c[index];
      if (index[num_dimensions]>0)
	{
	  const BasicCoordinate<num_dimensions, int> related_index = 
	    modulo(sizes-index, sizes);
	  result[related_index] = std::conj(c[index]);
	}
    }
  while(next(index, c));
  return result;
}


/*****************************************************************
 * INSTANTIATIONS
 * add any you need
 ******************************************************************/

// note: instantiate the highest dimension you need. That will do all lower dimensions
template
void 
fourier<>(Array<3,std::complex<float> >& c, const int sign);

template
void
fourier<>(Array<1,std::complex<float> >& c, const int sign);

template
void 
fourier<>(VectorWithOffset<std::complex<float> >& c, const int sign);

#define INSTANTIATE(d,type) \
 template \
 Array<d,std::complex<type> > \
 fourier_for_real_data<>(const Array<d,type>& v, const int sign); \
 template  \
 Array<d,type> \
  inverse_fourier_for_real_data_corrupting_input<>(Array<d,std::complex<type> >& c, const int sign); \
 template  \
 Array<d,type> \
  inverse_fourier_for_real_data<>(const Array<d,std::complex<type> >& c, const int sign); \
 template \
 Array<d, std::complex<type> > \
 pos_frequencies_to_all<>(const Array<d, std::complex<type> >& c);

INSTANTIATE(1,float);
INSTANTIATE(2,float);
INSTANTIATE(3,float);
#undef INSTANTIATE

END_NAMESPACE_STIR
