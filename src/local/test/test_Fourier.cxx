#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include <complex>
#include "stir/stream.h"
#include "stir/round.h"
#include "local/stir/fft.h"
#include "stir/CPUTimer.h"
#include <iostream>
#include <algorithm>


using std::cin;
using std::cout;
using std::complex;
using std::swap;
using std::endl;

START_NAMESPACE_STIR

template <typename elemT>
void fourier(VectorWithOffset<elemT>& c, const int sign);
template <typename elemT>
void fourier_1d(VectorWithOffset<elemT>& c, const int sign);

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#define elemT1 complex<float>
#define elemT2 complex<float>
#endif

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT1, typename elemT2>
#endif
VectorWithOffset<elemT1>&
operator -= (VectorWithOffset<elemT1>& lhs, const VectorWithOffset<elemT2>& rhs)
{
  assert(lhs.get_min_index() == rhs.get_min_index());
  assert(lhs.get_max_index() == rhs.get_max_index());
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  VectorWithOffset<elemT1>::const_iterator iter2= rhs.begin();
  while (iter1!= lhs.end())
    *iter1++ -= *iter2++;
  return lhs;
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT1
#undef elemT2
#define elemT1 VectorWithOffset<complex<float> >
#define elemT2 VectorWithOffset<complex<float> >
VectorWithOffset<elemT1>&
operator -= (VectorWithOffset<elemT1>& lhs, const VectorWithOffset<elemT2>& rhs)
{
  assert(lhs.get_min_index() == rhs.get_min_index());
  assert(lhs.get_max_index() == rhs.get_max_index());
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  VectorWithOffset<elemT1>::const_iterator iter2= rhs.begin();
  while (iter1!= lhs.end())
    *iter1++ -= *iter2++;
  return lhs;
}

#undef elemT1
#undef elemT2
#define elemT1 complex<float>
#define elemT2 complex<float>

#endif


#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT1, typename elemT2>
#endif
VectorWithOffset<elemT1>&
operator += (VectorWithOffset<elemT1>& lhs, const VectorWithOffset<elemT2>& rhs)
{
  assert(lhs.get_min_index() == rhs.get_min_index());
  assert(lhs.get_max_index() == rhs.get_max_index());
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  VectorWithOffset<elemT1>::const_iterator iter2= rhs.begin();
  while (iter1!= lhs.end())
    *iter1++ += *iter2++;
  return lhs;
}


#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT2
#define elemT2 float
#undef elemT1
#define elemT1 complex<float>
#endif

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT1, typename elemT2>
#endif
VectorWithOffset<elemT1>&
operator /= (VectorWithOffset<elemT1>& lhs, const elemT2& rhs)
{
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ /= rhs;
  return lhs;
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT2
#define elemT2 float
#undef elemT1
#define elemT1 float
VectorWithOffset<elemT1>&
operator /= (VectorWithOffset<elemT1>& lhs, const elemT2& rhs)
{
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ /= rhs;
  return lhs;
}
#undef elemT2
#define elemT2 float
#undef elemT1
#define elemT1 VectorWithOffset<complex<float> >
VectorWithOffset<elemT1>&
operator /= (VectorWithOffset<elemT1>& lhs, const elemT2& rhs)
{
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ /= rhs;
  return lhs;
}
#endif

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT2
#define elemT2 float
#undef elemT1
#define elemT1 complex<float>
#endif

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT1, typename elemT2>
#endif
VectorWithOffset<elemT1>&
operator *= (VectorWithOffset<elemT1>& lhs, const elemT2& rhs)
{
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ *= rhs;
  return lhs;
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT2
#define elemT2 complex<float>
VectorWithOffset<elemT1>&
operator *= (VectorWithOffset<elemT1>& lhs, const elemT2& rhs)
{
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ *= rhs;
  return lhs;
}
#endif


#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT1, typename elemT2>
#endif
VectorWithOffset<elemT1>&
operator += (VectorWithOffset<elemT1>& lhs, const elemT2& rhs)
{
  VectorWithOffset<elemT1>::iterator iter1= lhs.begin();
  while (iter1!= lhs.end())
    *iter1++ *= rhs;
  return lhs;
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT1
#undef elemT2
#endif

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT>
#else
#define elemT float
#endif
inline elemT 
norm (const VectorWithOffset<complex<elemT> > & v)
{
  elemT res=0;
  for (VectorWithOffset<complex<elemT> >::const_iterator iter= v.begin(); iter != v.end(); ++iter)
    res+= square(iter->real())+square(iter->imag());
  return sqrt(res);
}
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT
#endif

template <typename elemT>
inline elemT 
norm (const VectorWithOffset<VectorWithOffset<complex<elemT> > > & v)
{
  elemT res=0;
  for (VectorWithOffset<VectorWithOffset<complex<elemT> > >::const_iterator iter= v.begin(); iter != v.end(); ++iter)
    res+= square(norm(*iter));
  return sqrt(res);
}

template <typename T>
void bitreversal(VectorWithOffset<T>& data)
{
  const int n=data.get_length();
  int   j=1;
  for (int i=0;i<n;++i) {
    if (j/2 > i) {
      swap(data[j/2],data[i]);
    }
    int m=n;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
}

// exparray[k][i] = exp(i*_PI/pow(2,k))
static   VectorWithOffset<VectorWithOffset<complex<float> > > exparray;

void init_exparray(const int k, const int pow2k)
{
  if (exparray.get_max_index() >= k && exparray[k].get_length()>0)
    return;

  if (exparray.get_max_index() <k)
    exparray.grow(0, k);
  exparray[k].grow(0,pow2k-1);
  for (int i=0; i< pow2k; ++i)
    exparray[k][i]= std::exp(complex<float>(0, (i*_PI)/pow2k));
}

// expminarray[k][i] = exp(-i*_PI/pow(2,k))
static   VectorWithOffset<VectorWithOffset<complex<float> > > expminarray;

void init_expminarray(const int k, const int pow2k)
{
  if (expminarray.get_max_index() >= k && expminarray[k].get_length()>0)
    return;

  if (expminarray.get_max_index() <k)
    expminarray.grow(0, k);
  expminarray[k].grow(0,pow2k-1);
  for (int i=0; i< pow2k; ++i)
    expminarray[k][i]= std::exp(complex<float>(0, -(i*_PI)/pow2k));
}

#if 0
template <typename elemT>
void fourier(VectorWithOffset<VectorWithOffset<elemT> >& c, const int sign = 1)
{
  fourier_1d(c, sign);
  for (VectorWithOffset<VectorWithOffset<elemT> >::iterator iter = c.begin();
       iter != c.end();
       ++iter)
    fourier(*iter, sign);
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT>
void fourier(VectorWithOffset<elemT>& c, const int sign = 1)
{
  fourier_1d(c, sign);
}
#else
void fourier(VectorWithOffset<complex<float> >& c, const int sign)
{
  fourier_1d(c, sign);
}

void fourier(VectorWithOffset<complex<double> >& c, const int sign)
{
  fourier_1d(c, sign);
}
#endif

#else

template <typename elemT>
struct fourier_auxiliary
{
static  
void fourier(VectorWithOffset<elemT >& c, const int sign = 1);
};



#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT>
struct fourier_auxiliary<VectorWithOffset<elemT> >
{
  static void 
fourier(VectorWithOffset<VectorWithOffset<elemT> >& c, const int sign)
{
  fourier_1d(c, sign);
  for (VectorWithOffset<VectorWithOffset<elemT> >::iterator iter = c.begin();
       iter != c.end();
       ++iter)
         stir::fourier(*iter, sign);
}
};

template <typename elemT>
void 
fourier_auxiliary<elemT>::
fourier(VectorWithOffset<elemT>& c, const int sign = 1)
{
  fourier_1d(c, sign);
}
#else

template <typename elemT>
void 
fourier_auxiliary<elemT>::
fourier(VectorWithOffset<elemT>& c, const int sign)
{
  fourier_1d(c, sign);
  for (VectorWithOffset<elemT>::iterator iter = c.begin();
       iter != c.end();
       ++iter)
         stir::fourier(*iter, sign);
}

void 
fourier_auxiliary<complex<float> >::
fourier(VectorWithOffset<complex<float> >& c, const int sign)
{
  fourier_1d(c, sign);
}

void 
fourier_auxiliary<complex<double> >::
fourier(VectorWithOffset<complex<double> >& c, const int sign)
{
  fourier_1d(c, sign);
}
#endif

template <typename elemT>
void 
fourier(VectorWithOffset<elemT>& c, const int sign = 1)
{
  fourier_auxiliary<elemT>::fourier(c,sign);
}



#endif

// elemT has to be such that you can operator*=(elemT&, complex<float>) exist, i.e. 
// it has to be a complex type, or potentially a (multi-dimensional) array of complex numbers
template <typename elemT>
void fourier_1d(VectorWithOffset<elemT>& c, const int sign)
{
  if (c.get_length()==0) return;
  assert(c.get_min_index()==0);
  assert(sign==1 || sign ==-1);
  bitreversal(c);
  const int nn=stir::round(log(static_cast<double>(c.get_length()))/log(2.));
  if (c.get_length()!= stir::round(pow(2,nn)))
    error ("Fourier called with array length %d which is not 2^%d\n", c.get_length(), nn);


  int k=0;
  int pow2k = 1; // will be updated to be round(pow(2,k))
  const int pow2nn=c.get_length(); // ==round(pow(2,nn)); 
  for (; k<nn; ++k, pow2k*=2)
  {
    if (sign==1)
      init_exparray(k,pow2k);
    else
      init_expminarray(k,pow2k);
    for (int j=0; j< pow2nn;j+= pow2k*2) 
      for (int i=0; i< pow2k; ++i)
      {
        elemT& c1= c[i + j];
        elemT& c2= c[i + j + pow2k];
        
        const elemT t1 = c1; 
        //const complex<float> t2 = 
        //  sign==1? c2*exparray[k][i] : c2*expminarray[k][i];
        //c1 = t1+t2; c2 = t1-t2;
        c2 *= sign==1? exparray[k][i] : expminarray[k][i];
        c1 += c2;
        c2 *= -1;
        c2 += t1;
      }
  }
}

void inline inverse_fourier(VectorWithOffset<complex<float> >& c, const int sign = 1)
{
  fourier(c,-sign);
  c /= c.get_length();
}
//TODO multidim inverse_fourier

const int  FORWARDFFT=1;
const int INVERSEFFT=-1;


template <typename elemT>
void discrete_fourier_transform(VectorWithOffset<elemT>&data, int isign)
{
  const unsigned int nn = data.get_length()/2;
  const double TPI=2*_PI;
  unsigned int n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  elemT tempr,tempi;
  n=nn << 1;
  j=1;
  for (i=1;i<n;i+=2) {
    if (j > i) {
      swap(data[j],data[i]);
      swap(data[j+1],data[i+1]);
    }
    m=n >> 1;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax=2;
  while (n > mmax) {
    istep=mmax << 1;
    theta=isign*(TPI/mmax);
    wtemp=sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.0;
    wi=0.0;
    for (m=1;m<mmax;m+=2) {
      for (i=m;i<=n;i+=istep) {
        j=i+mmax;
        tempr=wr*data[j]-wi*data[j+1];
        tempi=wr*data[j+1]+wi*data[j];
        data[j]=data[i]-tempr;
        data[j+1]=data[i+1]-tempi;
        data[i] += tempr;
        data[i+1] += tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }
}


END_NAMESPACE_STIR

using namespace stir;

int main()
{
  
  int repeat=0;
  cout << "Enter repeat:";
  cin >> repeat;
  {

  VectorWithOffset<complex<float> > c;
#if 0
  cin >> c;
#else
  { 
    int length=0;
    cout << "Enter length of array:";
    cin >> length;
    c.grow(0,length-1);
    for (int i=0; i<c.get_length(); ++i)
      c[i]= complex<float>(rand(), rand());
  }
#endif
  VectorWithOffset<complex<float> > c_copy = c;
  VectorWithOffset<float> nr_data(1,2*c.get_length());
  for (int i=0; i<c.get_length(); ++i)
  {
    nr_data[2*i+1] = c[i].real();
    nr_data[2*i+2] = c[i].imag();
  }

  {
    CPUTimer timer;
    timer.start();
    for (int i=repeat; i; --i)
    {fourier(c); c/=sqrt(static_cast<float>(c.get_length()));}
    timer.stop();
    
    cout << "fourier " << timer.value() << '\n';
    
    timer.reset();
    timer.start();
    for (int i=repeat; i; --i)
    {discrete_fourier_transform(nr_data, FORWARDFFT); nr_data/=sqrt(static_cast<float>(c.get_length()));}
    timer.stop();
    cout << "NR " << timer.value() << '\n';
  }

  VectorWithOffset<complex<float> > nr_c(0,c.get_length()-1);
  for (int i=0; i<c.get_length(); ++i)
  {
    nr_c[i]=complex<float>(nr_data[2*i+1], nr_data[2*i+2]);
  }

  nr_c -= c;

  cout << "\nResidual norm " <<  norm(nr_c)/norm(c) << std::endl;

  for (int i=0; i<c.get_length(); ++i)
  {
    nr_data[2*i+1] = c[i].real();
    nr_data[2*i+2] = c[i].imag();
  }

  {
    CPUTimer timer;
    timer.start();
    for (int i=repeat; i; --i)
    {inverse_fourier(c);c*=sqrt(c.get_length());}
    timer.stop();    
    cout << "inverse fourier " << timer.value() << '\n';
    timer.reset();
    timer.start();
    for (int i=repeat; i; --i)
    {discrete_fourier_transform(nr_data, INVERSEFFT);nr_data /= c.get_length();nr_data/=1/sqrt(c.get_length());}
    
    timer.stop();
    cout << "NR " << timer.value() << '\n';
  }
  for (int i=0; i<c.get_length(); ++i)
  {
    nr_c[i]=complex<float>(nr_data[2*i+1], nr_data[2*i+2]);
  }

  nr_c -= c;
  cout << "\nResidual norm " << norm(nr_c)/norm(c) << std::endl;

  c -= c_copy;
  cout << "\nResidual norm inverse " << norm(c)<<','<<norm(c_copy)<<','<<norm(c)/norm(c_copy) << std::endl;
}
  
  {
    VectorWithOffset<VectorWithOffset<complex<float> > >  c2d;
     //Array<2,complex<float> > c2d;
  { 
    int length=0;
    cout << "Enter number of rows of array:";
    cin >> length;
    int columns=0;
    cout << "Enter number of columns of array:";
    cin >> columns;
    c2d.grow(0,length-1);
    for (int i=0; i<c2d.get_length(); ++i)
    {
      c2d[i].grow(0,columns-1);
      for (int j=0; j<c2d[i].get_length(); ++j)
        c2d[i][j]= complex<float>(rand(), rand());
    }
  }
   
    VectorWithOffset<VectorWithOffset<complex<float> > >  nr_c2d(c2d);
    //Array<2,complex<float> > nr_c2d(c2d);
    Array<1,float> nr_data(1,2*c2d.get_length()*c2d[0].get_length());
    {
      VectorWithOffset<float>::iterator nr_iter = nr_data.begin_all();
      
#if 0
      Array<2,complex<float> >::const_full_iterator iter=
        c2d.begin_all();
      while(iter != c2d.end_all())
      {
        *nr_iter++ = iter->real();
        *nr_iter++ = iter->imag();
        ++iter;
      }
#else
      VectorWithOffset<VectorWithOffset<complex<float> > >::const_iterator iter = 
        c2d.begin();
      while(iter != c2d.end())
      {
        VectorWithOffset<complex<float> >::const_iterator row_iter = iter->begin();
        while(row_iter != iter->end())
        {
          *nr_iter++ = row_iter->real();
          *nr_iter++ = row_iter->imag();
          ++row_iter;
        }
        ++iter;
      }
#endif
    }
    {
      CPUTimer timer;
      timer.start();
      for (int i=repeat; i; --i)
      {fourier(c2d);c2d/=sqrt(c2d.get_length()*c2d[0].get_length());}
      timer.stop();
      cout  << "fourier 2D " << timer.value() << '\n';
    }
      CPUTimer timer;
    timer.reset();
    timer.start();
    Array<1,int> dims(1,2);
    dims[1]=c2d.get_length();
    dims[2]=c2d[0].get_length();
    for (int i=repeat; i; --i)
    {fourn(nr_data, dims, 2, 1);nr_data/=sqrt(c2d.get_length()*c2d[0].get_length());}
    timer.stop();
    cout << "NR " << timer.value() << '\n';

    {
      VectorWithOffset<float>::const_iterator nr_iter = nr_data.begin_all();
#if 0
      Array<2,complex<float> >::full_iterator iter=
        nr_c2d.begin_all();
      while(iter != nr_c2d.end_all())
      {
        *iter = complex<float>(*nr_iter, *(nr_iter+1));
        nr_iter+= 2;
        ++iter;
      }      
#else
      VectorWithOffset<VectorWithOffset<complex<float> > >::iterator iter = 
        nr_c2d.begin();
      while(iter != nr_c2d.end())
      {
        VectorWithOffset<complex<float> >::iterator row_iter = iter->begin();
        while(row_iter != iter->end())
        {
          *row_iter = complex<float>(*nr_iter, *(nr_iter+1));
          nr_iter+= 2;
          ++row_iter;
        }
        ++iter;
      }
#endif
    }

  nr_c2d -= c2d;

  cout << "\nResidual norm " << norm(nr_c2d)/norm(c2d) << std::endl;

  }

  return EXIT_SUCCESS;
}



