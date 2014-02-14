//
//

/*!
  \file 
  \ingroup tests
  \brief Tests for function in the DFT group

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include <complex>
#include "stir/stream.h"
#include "stir/round.h"
#include "local/stir/fft.h"
#include "stir/CPUTimer.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/numerics/norm.h"
#include "stir/numerics/stir_NumericalRecipes.h"
#include "stir/numerics/fourier.h"
#include <iostream>
#include <algorithm>


using std::cin;
using std::cout;
using std::endl;

START_NAMESPACE_STIR
#define DOARRAY

#ifdef DOARRAY
  typedef Array<1,std::complex<float> > ArrayC1;
  typedef Array<1,float> ArrayF1;
  typedef Array<2,float> ArrayF2;
  typedef Array<2,std::complex<float> > ArrayC2;
  typedef Array<3,float> ArrayF3;
  typedef Array<3,std::complex<float> > ArrayC3;
#else
  typedef VectorWithOffset<std::complex<float> > ArrayC1;
  typedef VectorWithOffset<float> ArrayF1;
  typedef VectorWithOffset<ArrayC1 > ArrayC2;
#endif



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
      std::swap(data[j],data[i]);
      std::swap(data[j+1],data[i+1]);
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


inline float rand1() 
{
  return 2*(rand()-RAND_MAX/2.F)/RAND_MAX;
}

END_NAMESPACE_STIR

using namespace stir;

int main()
{
  cout << "Sign: ";
  int sign;
  cin >> sign;
  
#if 1
  int repeat=1;
  cout << "Enter repeat:";
  cin >> repeat;
  {

    ArrayC1 c;
#if 0
    cin >> c;
#else
    { 
      int length=0;
      cout << "Enter length of array:";
      cin >> length;
      c.grow(0,length-1);
      for (int i=0; i<c.get_length(); ++i)
	c[i]= std::complex<float>(rand1(), rand1());
    }
#endif
    const ArrayC1 c_copy = c;
    ArrayF1 nr_data(1,2*c.get_length());
    stir_to_nr(c, nr_data);

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
    // cout << '\n' << c << nr_c;

    ArrayC1 nr_c(0,c.get_length()-1);
    nr_to_stir(nr_data, nr_c);
    nr_c -= c;

    {
      const float tmp = norm(nr_c.begin(), nr_c.end())/norm(c.begin(), c.end());
      cout << "\nResidual norm " <<  tmp << std::endl;
      if (tmp > .1)
	{
	  std::cout << "NR : " << nr_c
		    << "STIR " << c;
	}
    }
    stir_to_nr(c, nr_data);

    {
      CPUTimer timer;
      timer.start();
      for (int i=repeat; i; --i)
	{inverse_fourier(c);c*=sqrt(static_cast<float>(c.get_length()));}
      timer.stop();    
      cout << "inverse fourier " << timer.value() << '\n';
      timer.reset();
      timer.start();
      for (int i=repeat; i; --i)
	{discrete_fourier_transform(nr_data, INVERSEFFT);nr_data /= c.get_length();nr_data/=1/sqrt(static_cast<float>(c.get_length()));}
    
      timer.stop();
      cout << "NR " << timer.value() << '\n';
    }
    nr_to_stir(nr_data, nr_c);

    nr_c -= c;
    cout << "\nResidual norm " << norm(nr_c.begin(), nr_c.end())/norm(c.begin(), c.end()) << std::endl;

    c -= c_copy;
    cout << "\nResidual norm inverse (relative)" <</* norm(c.begin(), c.end())<<','<<norm(c_copy.begin(), c_copy.end())<<','<<*/norm(c.begin(), c.end())/norm(c_copy.begin(), c_copy.end()) << std::endl;
  }
  
  {

    ArrayC2 c2d;
    { 
      int length=0;
      cout << "Enter number of rows of array:";
      cin >> length;
      int columns=0;
      cout << "Enter number of columns of array:";
      cin >> columns;
#ifndef DOARRAY
      c2d.grow(0,length-1);
      for (int i=0; i<c2d.get_length(); ++i)
	{
	  c2d[i].grow(0,columns-1);
	  for (int j=0; j<c2d[i].get_length(); ++j)
	    c2d[i][j]= std::complex<float>(rand1(), rand1());
	}
#else
      c2d.grow(IndexRange2D(length, columns));
      for (ArrayC2::full_iterator iter= c2d.begin_all();
	   iter!=c2d.end_all();
	   ++iter)
	    *iter= std::complex<float>(rand1(), rand1());
#endif
    }

    //cout << c2d;

    ArrayC2 nr_c2d(c2d);
    Array<1,float> nr_data(1,2*c2d.get_length()*c2d[0].get_length());
    stir_to_nr(nr_c2d, nr_data);

    const float 
      normfactor =sqrt(static_cast<float>(c2d.get_length()*c2d[0].get_length()));
    {
      CPUTimer timer;
      timer.start();
      for (int i=repeat; i; --i)
	{fourier(c2d);
	c2d/= normfactor;}
      timer.stop();
      cout  << "fourier 2D " << timer.value() << '\n';
    }
    {
      CPUTimer timer;
      timer.reset();
      timer.start();
      Array<1,int> dims(1,2);
      dims[1]=c2d.get_length();
      dims[2]=c2d[0].get_length();
      for (int i=repeat; i; --i)
	{fourn(nr_data, dims, 2, 1);nr_data/=normfactor;}
      timer.stop();
      cout << "NR " << timer.value() << '\n';
    }

    //cout << '\n' << c2d << '\n' << nr_data;
    ArrayF1 nr_data_stir_fourier(1,2*c2d.get_length()*c2d[0].get_length());
    stir_to_nr(c2d, nr_data_stir_fourier);
    nr_data -= nr_data_stir_fourier;  


    cout << "\nResidual norm " 
	 << norm(nr_data.begin(), nr_data.end())/
      norm(nr_data_stir_fourier.begin(), nr_data_stir_fourier.end()) 
	 << std::endl;

  }
#endif

  // ********** REAL ************
  {
    ArrayF1 v;
    { 
      int length=0;
      cout << "Enter length of array:";
      cin >> length;
      v.grow(0,length-1);
      for (int i=0; i<v.get_length(); ++i)
	v[i]= rand1();
      v[0]=rand1();
    }
    ArrayC1 pos_frequencies =
      fourier_1d_for_real_data(v,sign);
    const ArrayC1 all_frequencies =
      pos_frequencies_to_all(pos_frequencies);

    ArrayC1 c(v.get_length());
    std::copy(v.begin(), v.end(), c.begin());
    fourier(c,sign);
    //cout << all_frequencies << c;
    c -= all_frequencies;
    cout << "\nReal FT Residual norm "  <<
      norm(c.begin(), c.end())/norm(v.begin(), v.end());

    ArrayF1 again_v =
      inverse_fourier_for_real_data(pos_frequencies,sign);
    //cout <<"\nv,test "<< v << again_v << again_v/v;
    again_v -= v;
    cout << "\ninverse Real FT Residual norm "  <<
      norm(again_v.begin(), again_v.end())/norm(v.begin(), v.end());

  }
  {
    ArrayF2 v;
    { 
      int length=0;
      cout << "Enter number of rows of array:";
      cin >> length;
      int columns=0;
      cout << "Enter number of columns of array:";
      cin >> columns;
      v.grow(IndexRange2D(length, columns));
      for (ArrayF2::full_iterator iter= v.begin_all();
	   iter!=v.end_all();
	   ++iter)
	*iter= rand1();
    }
    ArrayC2 pos_frequencies =
      fourier_for_real_data(v,sign);
    const ArrayC2 all_frequencies =
      pos_frequencies_to_all(pos_frequencies);

    ArrayC2 c(v.get_index_range());
    std::copy(v.begin_all(), v.end_all(), c.begin_all());
    fourier(c,sign);
    //cout << pos_frequencies;
    //cout << all_frequencies << c;
    //cout << '\n' << c-all_frequencies;
    c -= all_frequencies;
    cout << "\nReal FT Residual norm "  <<
      norm(c.begin_all(), c.end_all())/norm(v.begin_all(), v.end_all());

    ArrayF2 again_v =
      inverse_fourier_for_real_data(pos_frequencies,sign);
    //cout <<"\nv,test "<< v << again_v << again_v/v;
    again_v -= v;
    cout << "\ninverse Real FT Residual norm "  <<
      norm(again_v.begin_all(), again_v.end_all())/norm(v.begin_all(), v.end_all());

  }
  {
    ArrayF3 v;
    { 
      int length=0;
      cout << "Enter number of rows of array:";
      cin >> length;
      int columns=0;
      cout << "Enter number of columns of array:";
      cin >> columns;
      int planes=0;
      cout << "Enter number of planes of array:";
      cin >> planes;
      v.grow(IndexRange3D(planes,length, columns));
      for (ArrayF3::full_iterator iter= v.begin_all();
	   iter!=v.end_all();
	   ++iter)
	*iter= rand1();
    }
    ArrayC3 pos_frequencies =
      fourier_for_real_data(v,sign);
    const ArrayC3 all_frequencies =
      pos_frequencies_to_all(pos_frequencies);

    ArrayC3 c(v.get_index_range());
    std::copy(v.begin_all(), v.end_all(), c.begin_all());
    fourier(c,sign);
    //cout << pos_frequencies;
    //cout << all_frequencies << c;
    //cout << '\n' << c-all_frequencies;
    c -= all_frequencies;
    cout << "\nReal FT Residual norm "  <<
      norm(c.begin_all(), c.end_all())/norm(v.begin_all(), v.end_all());

    ArrayF3 again_v =
      inverse_fourier_for_real_data(pos_frequencies,sign);
    //cout <<"\nv,test "<< v << again_v << again_v/v;
    again_v -= v;
    cout << "\ninverse Real FT Residual norm "  <<
      norm(again_v.begin_all(), again_v.end_all())/norm(v.begin_all(), v.end_all());

  }
  return EXIT_SUCCESS;
}



