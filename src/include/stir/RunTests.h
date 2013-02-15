//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup buildblock 
  \brief defines the stir::RunTests class

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/

#include "stir/VectorWithOffset.h"
#include "stir/BasicCoordinate.h"
#include "stir/IndexRange.h"
#include "stir/stream.h"
#include <iostream>
#include <typeinfo>
#include <vector>
#include <complex>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

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
  types. For instance, when std::size_t is not one of <code>unsigned int</code>
  or <code>unsigned long</code>, the compilation will break.

  A potential solution for this would be to use member templates. It would
  also remove some code repetition. However, you then need 
  specialisation of member template, and a relatively recent compiler.
  We leave this for later.
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
  /*! \return if this test worked */
  bool check(const bool, const char * const str = "");

  //! Tests equality, see check()
  bool check_if_equal(const double a, const double b, char const * const str = "");
    bool check_if_equal(const short a, const short b, char const * const str = "");
    bool check_if_equal(const unsigned short a, const unsigned short b, char const * const str = "");
    bool check_if_equal(const int a, const int b, char const * const str = "");
    bool check_if_equal(const unsigned int a, const unsigned int b, char const * const str = "");
    bool check_if_equal(const long a, const long b, char const * const str = "");
    bool check_if_equal(const unsigned long a, const unsigned long b, char const * const str = "");
#ifdef BOOST_HAS_LONG_LONG
    bool check_if_equal(const long long a, const long long b, char const * const str = "");
    bool check_if_equal(const unsigned long long a, const unsigned long long b, char const * const str = "");
#endif
  // VC 6.0 needs definition of template members in the class def unfortunately.
  template <class T>
    bool check_if_equal(const std::complex<T> a, const std::complex<T> b, char const * const str = "")
    {
      return 
	check_if_equal(a.real(), b.real(), str) &&
	check_if_equal(a.imag(), b.imag(), str);
    }
  template <class T>
    bool check_if_equal(const VectorWithOffset<T>& t1, const VectorWithOffset<T>& t2, 
                        const char * const str = "")
  {
    if (t1.get_min_index() != t2.get_min_index() ||
        t1.get_max_index() != t2.get_max_index())
    {
      cerr << "Error: unequal ranges. " << str << endl;
      return everything_ok = false;
    }

    for (int i=t1.get_min_index(); i<= t1.get_max_index(); i++)
    {
      if(!check_if_equal(t1[i], t2[i], str))
      {
        cerr << "(at VectorWithOffset<" << typeid(T).name() << "> first mismatch at index " << i << ")\n";
        return false;
      }
    }
    return true;
  }
  // VC 6.0 needs definition of template members in the class def unfortunately.
  template <class T>
	  bool check_if_equal(const std::vector<T>& t1, const std::vector<T>& t2, 
                        const char * const str = "")
  {
    if (t1.size() != t2.size())
    {
      cerr << "Error: unequal ranges. " << str << endl;
      return everything_ok = false;
    }

	bool all_equal=true;
    for (unsigned int i=0; i< t1.size(); i++)
    {
      if(!check_if_equal(t1[i], t2[i], str))
      {
        cerr << "(at vector<" << typeid(T).name() << ">  mismatch at index " << i << ")\n";
        all_equal=false;
      }
    }
    return all_equal;
  }
  
  // VC 6.0 needs definition of template members in the class def unfortunately.
  template <int n>
    bool check_if_equal(const IndexRange<n>& t1, const IndexRange<n>& t2, 
                        const char * const str = "")
  {
    if (t1 != t2)
    {
      cerr << "Error: unequal ranges. " << str << endl;
      return everything_ok = false;
    }
    else
      return true;
  }

  //! Tests if the value is 0, see check()
  bool check_if_zero(const double a, const char * const str = "");
  bool check_if_zero(const short a, char const * const str = "");
  bool check_if_zero(const unsigned short a, char const * const str = "");
  bool check_if_zero(const int a, char const * const str = "");
  bool check_if_zero(const unsigned int a, char const * const str = "");
  bool check_if_zero(const long a, char const * const str = "");
  bool check_if_zero(const unsigned long a, char const * const str = "");
#ifdef BOOST_HAS_LONG_LONG
  bool check_if_zero(const long long a, char const * const str = "");
  bool check_if_zero(const unsigned long long a, char const * const str = "");
#endif

  // VC needs definition of template members in the class def unfortunately.
  template <class T>
    bool check_if_zero(const VectorWithOffset<T>& t, const char * const str = "")
  {    
    for (int i=t.get_min_index(); i<= t.get_max_index(); i++)
    {
      if(!check_if_zero(t[i], str))
      {
        cerr << "(at VectorWithOffset<" << typeid(T).name() << "> first mismatch at index " << i << ")\n";
        return false;
      }
    }
    return true;
  }
  
  template <int num_dimensions, class coordT> 
  bool 
  check_if_zero(const BasicCoordinate<num_dimensions, coordT>& a, const char * const str = "" )
  {
    if (norm(a) > tolerance)
    {
      cerr << "Error : Coordinate value is " << a<< " expected 0s. " << str  << endl;
      everything_ok = false;
      return false;
    }
    else
      return true;
  }

  template <int num_dimensions, class coordT> 
  bool 
  check_if_equal(const BasicCoordinate<num_dimensions, coordT>& a,
                 const BasicCoordinate<num_dimensions, coordT>& b,
                 const char * const str = "" )
  { 
    if (norm(a-b) > tolerance)
    {
      cerr << "Error : Unequal Coordinate value are " 
           << a << " and " << b << ". " << str  << endl;
      everything_ok = false;
      return false;
    }
    else
      return true;
  }

private:
  double tolerance;
  bool everything_ok;

  template<class T>
  bool
    check_if_equal_generic(const T& a, const T& b, char const * const str)
    {
      if (a != b)
        {
          cerr << "Error : unequal values are " << a << " and " << b 
               << ". " << str<< endl;
          everything_ok = false;
          return false;
        }
      else
        return true;
    }

  template <class T> 
    bool 
    check_if_zero_generic(T a, const char * const str)
    {
      if ((a!=static_cast<T>(0)))
        {
          cerr << "Error : 0 value is " << a << " ." << str  << endl;
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
    cerr << "\nAll tests ok !\n\n" << endl;
  else
    cerr << "\nEnd of tests. Please correct errors !\n\n" << endl;
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

bool RunTests::check(const bool result, const char * const str)
{
  if (!result)
  {
    cerr << "Error. " << str << endl;
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
RunTests::check_if_equal(const double a, const double b, char const * const str)
{
  if ((fabs(b)>tolerance && fabs(a/b-1) > tolerance)
      || (fabs(b)<=tolerance && fabs(a-b) > tolerance))
  {
    cerr << "Error : unequal values are " << a << " and " << b 
         << ". " << str<< endl;
    everything_ok = false;
    return false;
  }
  else
    return true;
}


bool RunTests::check_if_equal(const short a, const short b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned short a, const unsigned  short b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const int a, const int b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned int a, const unsigned int b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const long a, const long b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned long a, const unsigned long b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
#ifdef BOOST_HAS_LONG_LONG
bool RunTests::check_if_equal(const long long a, const long long b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
bool RunTests::check_if_equal(const unsigned long long a, const unsigned long long b, char const * const str)
{  return this->check_if_equal_generic(a,b,str);}
#endif

bool RunTests::check_if_zero(const short a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned short a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const int a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned int a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const long a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned long a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
#ifdef BOOST_HAS_LONG_LONG
bool RunTests::check_if_zero(const long long a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
bool RunTests::check_if_zero(const unsigned long long a, char const * const str)
{  return this->check_if_zero_generic(a,str);}
#endif

bool 
RunTests::check_if_zero(const double a, const char * const str )
{
  if (fabs(a) > tolerance)
  {
    cerr << "Error : 0 value is " << a << " ." << str<< endl;
    everything_ok = false;
    return false;
  }
  else
    return true;
}

END_NAMESPACE_STIR
