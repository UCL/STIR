//
// $Id$: $Date$
//

/*!
  \file 
  \ingroup buildblock 
  \brief defines the RunTests class

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$
*/

#include "VectorWithOffset.h"
#include "BasicCoordinate.h"
#include "stream.h"
#include <iostream>
#ifndef OLDDESIGN
#include <typeinfo>
#endif

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_TOMO

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
*/

// some of this could be written nicer using more member templates
// handling the int case separately would require specialisation of member templates though

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
  bool check_if_equal(const int a, const int b, char const * const str = "");
  // VC needs definition of template members in the class def unfortunately.
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
  

  //! Tests if the value is 0, see check()
  bool check_if_zero(const int a, const char * const str = "");
  bool check_if_zero(const double a, const char * const str = "");

  // VC needs definition of template members in the class def unfortunately.
  template <class T>
    bool check_if_zero(const VectorWithOffset<T>& t, const char * const str = "")
  {    
    for (int i=t.get_min_index(); i<= t.get_max_index(); i++)
    {
      if(!check_if_zero(t[i], str))
      {
#ifndef OLDDESIGN
        cerr << "(at VectorWithOffset<" << typeid(T).name() << "> first mismatch at index " << i << ")\n";
#endif
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
  
};

END_NAMESPACE_TOMO


START_NAMESPACE_TOMO

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

bool
RunTests::check_if_equal(const int a, const int b, char const * const str)
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

bool 
RunTests::check_if_zero(const int a, const char * const str)
{
  if ((a!=0))
  {
    cerr << "Error : 0 value is " << a << " ." << str  << endl;
    everything_ok = false;
    return false;
  }
  else
    return true;
}

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

END_NAMESPACE_TOMO
