/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2005, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
    Copyright (C) 2013, University College London
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
  \ingroup test
  \ingroup buildblock 
  \brief defines the stir::RunTests class

  \author Kris Thielemans
  \author PARAPET project
*/

#include "stir/VectorWithOffset.h"
#include "stir/BasicCoordinate.h"
#include "stir/IndexRange.h"
#include "stir/stream.h"
#include <iostream>
#include <typeinfo>
#include <vector>
#include <complex>
#include <string>

START_NAMESPACE_STIR

/*!
  \brief A base class for making test classes
  \ingroup test
  \ingroup buildblock

  With a derived class, an application could look like
\code
int main(int argc, char **argv, char **env)
{
  SomeTests tests(argc,argv,env);
  tests.run_tests();
  return tests.main_return_value();
}
\endcode

  \par Implementation notes

  At present, the various check* functions are overloaded on most basic
  types, and in some STIR specific types such as VectorWithOffset. 
  This creates problems when the arguments are not exactly of one of those
  types, as then the compilation might break.

  A potential solution for this would be to use member templates and the
  addition of a 'generic' function as a template. However, then that
  function would take precedence for other cases which are currently resolved
  by overloading (such as when using stir::Array which is currently handled via 
  stir::VectorWithOffset). 

  In summary, if you need to check a different type that's not provided for,
  either use check(), add an appropriate function to your derived class, or
  if useful for other cases to this class of course.
*/
class RunTests
{
public:
  //! Default constructor
  explicit RunTests( const double tolerance = 1E-4);

  //! Destructor, outputs a diagnostic message
  virtual ~RunTests();

  //! Function (to be overloaded) which does the actual tests.
  /*! This function is expected to do a series of calls to check(), check_if_equal() etc. */
  virtual void run_tests() = 0;
  
  //! Returns if all checks were fine upto now.
  bool is_everything_ok() const;

  //! Handy return value for a main() function
  /*! Returns EXIT_SUCCESS if is_everything_ok(), EXIT_FAILURE otherwise */
  int main_return_value() const;
  
  //! Set value used in floating point comparisons (see check_* functions)
  void set_tolerance(const double tolerance);

  //! Get value used in floating point comparisons (see check_* functions)
  double get_tolerance() const;  

  //! Tests if true, str can be used to tell what you are testing
  /*! \return if this test worked.
   If the test failed, writes diagnostic to \c std::cerr (including value of str) and 
   modifies internal variable \c everything_ok.
  */
  bool check(const bool, const std::string& str = "");

  /*! \name Test equality
    float/double/complex numbers are compared using \c tolerance.
    \see check()
  */
  //@{
  bool check_if_equal(const double a, const double b, const std::string& str = "");
  // Note: due to current C++ overloading rules, we need copies for every integer type
  bool check_if_equal(const short a, const short b, const std::string& str = "");
  bool check_if_equal(const unsigned short a, const unsigned short b, const std::string& str = "");
  bool check_if_equal(const int a, const int b, const std::string& str = "");
  bool check_if_equal(const unsigned int a, const unsigned int b, const std::string& str = "");
  bool check_if_equal(const long a, const long b, const std::string& str = "");
  bool check_if_equal(const unsigned long a, const unsigned long b, const std::string& str = "");
#ifdef BOOST_HAS_LONG_LONG
  // necessary to cope with std::size_t for instance (on 64 bit systems)
  bool check_if_equal(const long long a, const long long b, const std::string& str = "");
  bool check_if_equal(const unsigned long long a, const unsigned long long b, const std::string& str = "");
#endif
  // VC 6.0 needs definition of template members in the class def unfortunately.
  //! check equality by calling check_if_equal on real and imaginary parts
  template <class T>
    bool check_if_equal(const std::complex<T> a, const std::complex<T> b, const std::string& str = "")
    {
      return 
	check_if_equal(a.real(), b.real(), str) &&
	check_if_equal(a.imag(), b.imag(), str);
    }
  //! check equality by comparing ranges and calling check_if_equal on all elements
  template <class T>
    bool check_if_equal(const VectorWithOffset<T>& t1, const VectorWithOffset<T>& t2, 
                        const std::string& str = "")
  {
    if (t1.get_min_index() != t2.get_min_index() ||
        t1.get_max_index() != t2.get_max_index())
    {
      std::cerr << "Error: unequal ranges. " << str << std::endl;
      return everything_ok = false;
    }

    for (int i=t1.get_min_index(); i<= t1.get_max_index(); i++)
    {
      if(!check_if_equal(t1[i], t2[i], str))
      {
        std::cerr << "(at VectorWithOffset<" << typeid(T).name() << "> first mismatch at index " << i << ")\n";
        return false;
      }
    }
    return true;
  }
  // VC 6.0 needs definition of template members in the class def unfortunately.
  //! check equality by comparing size and calling check_if_equal on all elements
  template <class T>
	  bool check_if_equal(const std::vector<T>& t1, const std::vector<T>& t2, 
                        const std::string& str = "")
  {
    if (t1.size() != t2.size())
    {
      std::cerr << "Error: unequal ranges. " << str << std::endl;
      return everything_ok = false;
    }

	bool all_equal=true;
    for (unsigned int i=0; i< t1.size(); i++)
    {
      if(!check_if_equal(t1[i], t2[i], str))
      {
        std::cerr << "(at vector<" << typeid(T).name() << ">  mismatch at index " << i << ")\n";
        all_equal=false;
      }
    }
    return all_equal;
  }
  
  // VC 6.0 needs definition of template members in the class def unfortunately.
  template <int n>
    bool check_if_equal(const IndexRange<n>& t1, const IndexRange<n>& t2, 
                        const std::string& str = "")
  {
    if (t1 != t2)
    {
      std::cerr << "Error: unequal ranges. " << str << std::endl;
      return everything_ok = false;
    }
    else
      return true;
  }
  //! check equality by comparing norm(a-b) with tolerance
  template <int num_dimensions, class coordT> 
  bool 
  check_if_equal(const BasicCoordinate<num_dimensions, coordT>& a,
                 const BasicCoordinate<num_dimensions, coordT>& b,
                 const std::string& str = "" )
  { 
    if (norm(a-b) > tolerance)
    {
      std::cerr << "Error : Unequal Coordinate value are " 
           << a << " and " << b << ". " << str  << std::endl;
      everything_ok = false;
      return false;
    }
    else
      return true;
  }
  //@}

  /*! \name Test if a value is zero
    float/double/complex numbers are compared using \c tolerance.
    \see check()
  */
  //@{
  bool check_if_zero(const double a, const std::string& str = "");
  bool check_if_zero(const short a, const std::string& str = "");
  bool check_if_zero(const unsigned short a, const std::string& str = "");
  bool check_if_zero(const int a, const std::string& str = "");
  bool check_if_zero(const unsigned int a, const std::string& str = "");
  bool check_if_zero(const long a, const std::string& str = "");
  bool check_if_zero(const unsigned long a, const std::string& str = "");
#ifdef BOOST_HAS_LONG_LONG
  bool check_if_zero(const long long a, const std::string& str = "");
  bool check_if_zero(const unsigned long long a, const std::string& str = "");
#endif

  // VC needs definition of template members in the class def unfortunately.
  //! use check_if_zero on all elements
  template <class T>
    bool check_if_zero(const VectorWithOffset<T>& t, const std::string& str = "")
  {    
    for (int i=t.get_min_index(); i<= t.get_max_index(); i++)
    {
      if(!check_if_zero(t[i], str))
      {
        std::cerr << "(at VectorWithOffset<" << typeid(T).name() << "> first mismatch at index " << i << ")\n";
        return false;
      }
    }
    return true;
  }
  
  //! compare norm with tolerance
  template <int num_dimensions, class coordT> 
  bool 
  check_if_zero(const BasicCoordinate<num_dimensions, coordT>& a, const std::string& str = "" )
  {
    if (norm(a) > tolerance)
    {
      std::cerr << "Error : Coordinate value is " << a<< " expected 0s. " << str  << std::endl;
      everything_ok = false;
      return false;
    }
    else
      return true;
  }
  //@}


protected:
  //! tolerance for comparisons with real values
  double tolerance;
  //! variable storing current status
  bool everything_ok;

  //! function that is called by some check_if_equal implementations. It just uses operator!=
  template<class T>
  bool
    check_if_equal_generic(const T& a, const T& b, const std::string& str)
    {
      if (a != b)
        {
          std::cerr << "Error : unequal values are " << a << " and " << b 
               << ". " << str<< std::endl;
          everything_ok = false;
          return false;
        }
      else
        return true;
    }

  //! function that is called by some check_if_zero implementations. It just uses operator!=
  template <class T> 
    bool 
    check_if_zero_generic(T a, const std::string& str)
    {
      if ((a!=static_cast<T>(0)))
        {
          std::cerr << "Error : 0 value is " << a << " ." << str  << std::endl;
          everything_ok = false;
          return false;
        }
      else
        return true;
    }
  
};

END_NAMESPACE_STIR


START_NAMESPACE_STIR

RunTests::RunTests(const double tolerance)
  : tolerance(tolerance), everything_ok(true)
{}

RunTests::~RunTests()
{
#if defined(_DEBUG) && defined(__MSL__)
  DebugNewReportLeaks();
#endif   

  if (everything_ok)
    std::cerr << "\nAll tests ok !\n\n" << std::endl;
  else
    std::cerr << "\nEnd of tests. Please correct errors !\n\n" << std::endl;
}

bool RunTests::is_everything_ok() const
{ return everything_ok ; }

int RunTests::main_return_value() const
{ return everything_ok ? EXIT_SUCCESS : EXIT_FAILURE; }

void RunTests::set_tolerance(const double tolerance_v)
{
  tolerance = tolerance_v;
}

double RunTests::get_tolerance() const
{ return tolerance; }

bool RunTests::check(const bool result, const std::string& str)
{
  if (!result)
  {
    std::cerr << "Error. " << str << std::endl;
    everything_ok = false;
  }
  return result;
}


/*! tolerance is used to account for floating point rounding error. Equality is checked using
\code
((fabs(b)>tolerance && fabs(a/b-1) > tolerance)
      || (fabs(b)<=tolerance && fabs(a-b) > tolerance))
\endcode
*/
bool
RunTests::check_if_equal(const double a, const double b, const std::string& str)
{
  if ((fabs(b)>tolerance && fabs(a/b-1) > tolerance)
      || (fabs(b)<=tolerance && fabs(a-b) > tolerance))
  {
    std::cerr << "Error : unequal values are " << a << " and " << b 
         << ". " << str<< std::endl;
    everything_ok = false;
    return false;
  }
  else
    return true;
}


bool RunTests::check_if_equal(const short a, const short b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned short a, const unsigned  short b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const int a, const int b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned int a, const unsigned int b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const long a, const long b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned long a, const unsigned long b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
#ifdef BOOST_HAS_LONG_LONG
bool RunTests::check_if_equal(const long long a, const long long b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned long long a, const unsigned long long b, const std::string& str)
{  return this->check_if_equal_generic(a,b,str);}
#endif

bool RunTests::check_if_zero(const short a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned short a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const int a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned int a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const long a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned long a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
#ifdef BOOST_HAS_LONG_LONG
bool RunTests::check_if_zero(const long long a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned long long a, const std::string& str)
{  return this->check_if_zero_generic(a,str);}
#endif

/*! check if \code fabs(a)<=tolerance \endcode */
bool 
RunTests::check_if_zero(const double a, const std::string& str )
{
  if (fabs(a) > tolerance)
  {
    std::cerr << "Error : 0 value is " << a << " ." << str<< std::endl;
    everything_ok = false;
    return false;
  }
  else
    return true;
}

END_NAMESPACE_STIR
