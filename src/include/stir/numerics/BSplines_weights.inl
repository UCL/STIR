//
//
/*
    Copyright (C) 2005 - 2009-10-27, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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
  \ingroup BSpline
  \brief Implementation of the B-Splines Interpolation 

  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/


START_NAMESPACE_STIR
namespace BSpline 
{

  template <class posT>
  class PieceWiseFunction
  {
  public:
    typedef posT result_type;

    virtual ~PieceWiseFunction() {}

    virtual posT kernel_length_left() const = 0;
    virtual int kernel_total_length() const = 0;
    posT kernel_length_right() const 
    { return -this->kernel_length_left() + this->kernel_total_length(); }
    
    virtual posT function_piece(const posT x, int p) const = 0;
    virtual posT derivative_piece(const posT x, int p) const = 0;
    virtual int find_piece(const posT x) const = 0;
    virtual int find_highest_piece() const = 0;
    posT function(const posT x) const
    {
      return this->function_piece(x, find_piece(x));
    }
    posT derivative(const posT x) const
    {
      return this->derivative_piece(x, find_piece(x));
    }
    posT operator()(const posT x) const
    {
      return this->function(x);
    }
    posT operator()(const posT p, const BSplineType)
    {
      return this->function(p);
    }

  };

  template <BSplineType bspline_type, class posT>
  class BSplineFunction : public PieceWiseFunction<posT>
  {};

  template <class posT>
  class BSplineFunction<near_n, posT>: public PieceWiseFunction<posT>
  {
  public:
    BSplineFunction() {}
    posT kernel_length_left() const { return static_cast<posT>(.5); }
    BOOST_STATIC_CONSTANT(int, kernel_total_length_value=1);
    int kernel_total_length() const  { return kernel_total_length_value; }
    posT function_piece(const posT x, int p) const
    {
      switch (p)
        {
        case 0:
          return static_cast<posT>(1);
          // todo probably should add another piece containing a just .5, now returns 0
        default:
          return 0;
        }
    }
    posT derivative_piece(const posT, int ) const
    {
      return 0;
    }
    
    int find_abs_piece(const posT absx) const
    {
      return static_cast<int>(absx+.5);
    }
    int find_piece(const posT x) const
    {
      const int abs_p = this->find_abs_piece(fabs(x));
      if (x>0)
        return abs_p;
      else
        return -abs_p;
    }
    int find_highest_piece() const
    {
      return 0;
    }
  };

  template <class posT>
  class BSplineFunction<linear, posT>: public PieceWiseFunction<posT>
  {
  public:
    BSplineFunction() {}
    posT kernel_length_left() const { return static_cast<posT>(1); }
    BOOST_STATIC_CONSTANT(int, kernel_total_length_value=2);
    int kernel_total_length() const  { return kernel_total_length_value; }
    posT function_piece(const posT x, int p) const
    {
      switch (p)
        {
        case 0:
          return 1-x;
        case -1:
          return 1+x;
        default:
          return 0;
        }
    }
    posT derivative_piece(const posT x, int p) const
    {
      switch (p)
        {
        case 0:
          return -1;
        case -1:
          return 1;
        default:
          return 0;
        }
    }
    int find_piece(const posT x) const
    {
      return static_cast<int>(floor(x));
    }

    int find_highest_piece() const
    {
      return 0;
    }
  };

  template <class posT>
  class BSplineFunction<quadratic, posT>: public PieceWiseFunction<posT>
  {
  public:
    BSplineFunction() {}
    posT kernel_length_left() const { return static_cast<posT>(1.5); }
    BOOST_STATIC_CONSTANT(int, kernel_total_length_value=3);
    int kernel_total_length() const  { return kernel_total_length_value; }
    posT function_piece(const posT x, int p) const
    {
      switch (std::abs(p))
        {
        case 0:
          return static_cast<posT>(3)/4 - square(x);
        case 1:
          {
            const posT absx = std::fabs(x);
            return square(2*absx - 3)/8;
          }
        default:
          return 0;
        }
    }
    posT derivative_piece(const posT x, int p) const
    {
      switch (std::abs(p))
        {
        case 0:
          return -2*x;
        case 1:
          {
            const int sign= x>0?1:-1;
            return x - sign*static_cast<posT>(1.5);
          }
        default:
          return 0;
        }
    }
  private:
    int find_abs_piece(const posT absx) const
    {
#if 1
      return static_cast<int>(absx+.5);
#else
      // can use this if guaranteed never out of range
      if (absx<=.5)
        return 0;
      else if (absx<=1.5)
        return 1;
      else 
        return 2;
#endif

    }
  public:
    int find_piece(const posT x) const
    {
      const int abs_p = this->find_abs_piece(fabs(x));
      if (x>0)
        return abs_p;
      else
        return -abs_p;
    }
    int find_highest_piece() const
    {
      return 1;
    }

#if 0
    posT function(const posT x) const
    {
      const posT absx = fabs(x);
      // note: gcc inlines next function call, even getting rid of switch
      // so, this is faster than using virtual functions etc.
      // however, ugly code repetition, so we can't be bothered, as this is hardly called anyway
      if (absx<=.5)
        return self_t::function_piece(absx,0);
      else if (absx<=1.5)
        return self_t::function_piece(absx,1);
      else
        return 0;
    }
#endif
  };

  template <class posT>
  class BSplineFunction<cubic, posT>: public PieceWiseFunction<posT>
  {
  public:
    BSplineFunction() {}
    posT kernel_length_left() const { return static_cast<posT>(2); }
    BOOST_STATIC_CONSTANT(int, kernel_total_length_value=4);
    int kernel_total_length() const  { return kernel_total_length_value; }
    posT function_piece(const posT x, int p) const
    {
      const posT absx = std::fabs(x);
      switch (p)
        {
        case 0:
        case -1:
          return 2./3. + (absx/2-1)*absx*absx;
        case 1:
        case -2:
          {
            const posT tmp=2-absx;
            return tmp*tmp*tmp/6;
          }
        default:
          return 0;
        }
    }
    posT derivative_piece(const posT x, int p) const
    {
      switch (p)
        {
        case 0:
          return x*(1.5*x-2);
        case -1:
          return -x*(1.5*x+2);
        case 1:
          return -square(x-2)/2;
        case -2:
          return square(x+2)/2;
        default:
          return 0;
        }
    }
    int find_piece(const posT x) const
    {
      return static_cast<int>(floor(x));
    }
    int find_highest_piece() const
    {
      return 1;
    }
  };


  template <class posT>
  class BSplineFunction<oMoms, posT>: public PieceWiseFunction<posT>
  {
  public: 
    BSplineFunction() {}
    posT kernel_length_left() const { return static_cast<posT>(2); }
    BOOST_STATIC_CONSTANT(int, kernel_total_length_value=4);
    int kernel_total_length() const  { return kernel_total_length_value; }
    posT function_piece(const posT x, int p) const
    {
      const posT absx = std::fabs(x);
      switch (p)
        {
        case 0:
        case -1:
          return 13./21. + absx*(1./14. +absx*(absx/2-1)) ;
        case 1:
        case -2:
          return 29./21. + absx*(-85./42. + absx*(1-absx/6)) ;          
        default:
          return 0;
        }
    }
    posT derivative_piece(const posT x, int p) const
    {
      const posT absx = std::fabs(x);
      const int sign=x>0?1:-1;
      switch (p)
        {
        case 0:
        case -1:
          return sign*(1/14. + absx*(-2+3*absx/2));
        case 1:
        case -2:
          return sign*(-85/42. + absx*(2-absx/2));
        default:
          return 0;
        }
    }
    int find_piece(const posT x) const
    {
      return static_cast<int>(floor(x));
    }
    int find_highest_piece() const
    {
      return 1;
    }
  };

  static const BSplineFunction<near_n, pos_type> near_n_BSpline_function;
  static const BSplineFunction<linear, pos_type> linear_BSpline_function;
  static const BSplineFunction<quadratic, pos_type> quadratic_BSpline_function;
  static const BSplineFunction<cubic, pos_type> cubic_BSpline_function;
  static const BSplineFunction<oMoms, pos_type> oMoms_BSpline_function;

  inline
  const PieceWiseFunction<pos_type>&
  bspline_function(BSplineType type)
  {
    switch (type)
      {
      case near_n:
        return near_n_BSpline_function;
      case linear:
        return linear_BSpline_function;
      case quadratic:
        return quadratic_BSpline_function;
      case cubic:
        return cubic_BSpline_function;
      case oMoms:
        return oMoms_BSpline_function;
      default:  
        std::cerr << "quartic,quantic to do\n";
        exit(EXIT_FAILURE);
        return cubic_BSpline_function;//WARNING WRONG
      }
  }

  template <typename posT>
  inline
  posT 
  cubic_BSplines_weight(const posT relative_position) 
  {     
    const posT abs_relative_position = fabs(relative_position);
    assert(abs_relative_position>=0);
    if (abs_relative_position<1)                
      return 2./3. + (0.5*abs_relative_position-1)*abs_relative_position*abs_relative_position;
    if (abs_relative_position>=2)
      return 0;
    const posT tmp=2-abs_relative_position;
    return tmp*tmp*tmp/6;
        
  }


  template <typename posT>
  inline
  posT 
  oMoms_weight(const posT relative_position) 
  {
    const posT abs_relative_position = fabs(relative_position);
    assert(abs_relative_position>=0);
    if (abs_relative_position>=2)
      return 0;
    if (abs_relative_position>=1)               
      return 29./21. + abs_relative_position*(-85./42. + 
                                              abs_relative_position*(1.-abs_relative_position/6)) ;             
    else 
      return 13./21. + abs_relative_position*(1./14. + 
                                              abs_relative_position*(0.5*abs_relative_position-1.)) ;
  }

  template <typename posT>
  inline
  posT 
  cubic_BSplines_1st_der_weight(const posT relative_position) 
  {
    const posT abs_relative_position = fabs(relative_position);
    if (abs_relative_position>=2)
      return 0;
    int sign = relative_position>0?1:-1;
    if (abs_relative_position>=1)               
      return -0.5*sign*(abs_relative_position-2)*(abs_relative_position-2);
    return sign*abs_relative_position*(1.5*abs_relative_position-2.);   
  }

  template <typename posT>
  posT 
  BSplines_1st_der_weight(const posT relative_position, const BSplineType spline_type) 
  {
    switch(spline_type)
      {
      case cubic:
        return cubic_BSplines_1st_der_weight(relative_position);        
      default:
        error("BSplines_1st_der_weight currently only implemented for cubic splines.");
        return 0;
        // TODO
      }
  }

  template <typename posT>
  posT 
  BSplines_weights(const posT relative_position, const BSplineType spline_type) 
  {
    //  double relative_position = rel_position;
    switch(spline_type)
      {
      case cubic:
        return cubic_BSplines_weight(relative_position);        
      case near_n:
        {               
          if (fabs(relative_position)<0.5)
            return 1;           
          if (fabs(relative_position)==0.5)
            return 0.5;         
          else return 0;
        }
      case linear:
        {
          if (fabs(relative_position)<1)
            return 1-fabs(relative_position);
          else
            return 0;
        }
      case quadratic:
        return  BSplineFunction<quadratic,posT>().function(relative_position);      
      case quintic:
        return  
          (pow(std::max(0.,-3. + relative_position),5) -  6*pow(std::max(0.,-2. + relative_position),5) +
           15*pow(std::max(0.,-1. + relative_position),5) - 20*pow(std::max(static_cast<posT>(0),relative_position),5) + 
           15*pow(std::max(0.,1. + relative_position),5) -   6*pow(std::max(0.,2. + relative_position),5) + 
           pow(std::max(0.,3. + relative_position),5))/ 120. ;          
      case oMoms:
        return oMoms_weight(relative_position); 
      default:
        error("Not implemented b-spline type");
        return -1000000;
      }
  }

} // end BSpline namespace

END_NAMESPACE_STIR

