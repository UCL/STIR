//
//

/*! 
  \file 
  \brief Filter classes (filter defined in Fourier space)
  \author Kris Thielemans
  \author Claire LABBE
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __FILTER_H__
#define  __FILTER_H__

#include "stir/TimedObject.h"
#include "stir/Array.h"
#include "local/stir/fft.h"
#include <string>


#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class Array;


/*!
  \brief Preliminary class for 1D filtering using FFTs

  \warning The filtering will be performed using the convlvC() function. See warnings in its documentation.
  \todo apply() members can't be const as they call TimedObject::start_timers()
*/
template <class T> class Filter1D : public TimedObject
 {
    // AZ public, as need to access length for parallel FBP3DRP code  (TODO)
// protected:
  public:
    //! Stores the filter in frequency space (will be private sometime)
    Array<1,float> filter; 
 public:
    Filter1D(const int length_v)
            : filter(1,length_v)
	{}
    Filter1D(const Array<1,T> &filter)
            : filter(filter)
	{}
   virtual ~Filter1D() {}

    //! Filters data (which has to be in the 'spatial' domain
    /*! data will be padded with zeroes to the same length as the stored filter before 
        filtering. This is done on a local copy of the data, such that the index range of 
        \a data is not modified.
    */
    inline void apply(Array<1,T> &data);
    //! Applies the filter on each 1D subarray
    inline void apply(Array<2,T> &data);
    //! Applies the filter on each 1D subarray
    inline void apply(Array<3,T> &data);

    
    virtual string parameter_info() const = 0;
};

// TODO can't be const due to start_timers()
template <class T> void Filter1D <T>::apply(Array<1,T> &data) //const
{
  start_timers();
  const int length = filter.get_length();
  if (length==0)
    return;
  assert(length>0);

  Array<1,T> Padded = data;
  Padded.set_offset(1);
  Padded.grow(1,length);
  convlvC(Padded, filter,length);

  Padded.set_offset(data.get_min_index());

  for (int j=data.get_min_index();j<=data.get_max_index();j++)
       data[j]= Padded[j];

  stop_timers();
}

template <class T> void   Filter1D <T>::apply(Array<2,T> &data)// const
{
    for (int i= data.get_min_index(); i <= data.get_max_index(); i++)
        apply(data[i]);
}

template <class T> void   Filter1D <T>::apply(Array<3,T> &data)// const
{
    for (int i= data.get_min_index(); i <= data.get_max_index(); i++)
        apply(data[i]);
}



/*!
  \brief 2-dimensional filters (filtering done by FFTs)

*/
template <class T> class Filter2D : public TimedObject
{
public:
  Filter2D(int height_v, int width_v)
    : height(height_v), width(width_v), filter(1,width_v*height_v)
    {}

   virtual ~Filter2D() {} 
  inline void apply(Array<1,T> &data) const;

  // TODO ???
  void padd_scale_filter(int height_proj, int width_proj);
    
  virtual string parameter_info() const = 0;

protected:
  int height; 
  int width; 

  //! Stores the filter in the 'funny' but efficient Numerical Recipes format
  Array<1,T> filter; 

};



  
template <class T> void Filter2D<T>::apply(Array<1,T> &data) const
{

  assert(data.get_length()  == 2*height*width);

  assert(data.get_min_index() == 1);

  int j,k;
  /*TODO copy lines from Filter_proj_Colsher for this convertion, and call apply with a Array<2,T>
  Array<1,T> data(1, 2*height*width);

  for (int h = 1, j=1; h <= data2d.height(); h++)
    for (int w=1; w <= data2d.get_width(); w++) {
      data[j++] = data2d[h][w];
    }
    */
  
  Array<1,int>  nn(1,2);
  nn[1] = height;
  nn[2] = width;
  fourn(data, nn, 2, 1);
  for (j = 1, k = 1; j < width * height + 1; j++) 
  {
    // KT 11/04/2000 normalise with total length
    // divide by width*height to take Num Recipes convention into account that the
    // 'inverse' Fourier transform needs scaling by the total number of data points.
    data[k++] *= filter[j]/(width * height);
    data[k++] *= filter[j]/(width * height);
  }
  fourn(data, nn, 2, -1);

};

END_NAMESPACE_STIR

#endif

