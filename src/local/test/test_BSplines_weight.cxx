
  
#include "stir/RunTests.h"
#include "local/stir/IR_filters.h"
//#include "local/stir/BSplines_coef.h"
#include <list>
#include <algorithm>


#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the BSplines_coef function.
*/
class BSplines_weightTests : public RunTests
{
public:
  BSplines_weightTests() 
  {}
  void run_tests();
private:
  //istream& in;
};


void BSplines_weightTests::run_tests()
{  
  cerr << "Testing BSplines_weight function..." << endl;

  set_tolerance(1.0);
 
  

  std::list<int> BSplinesWeightSTIRList, BSplinesWeightCorrectList;

  for(int i=3; i!=-2 ;--i)
	  BSplinesWeightList.push_back(BSplinesWeight(i));
  

	  BSplinesWeightCorrectList.push_back(10); //1
	  BSplinesWeightCorrectList.push_back(29); //2
	  BSplinesWeightCorrectList.push_back(56); //3
	  BSplinesWeightCorrectList.push_back(90); //4
	  BSplinesWeightCorrectList.push_back(130); //5
	  
	  {   		  

		  std::list<float>:: iterator cur_iter_stir_out= BSplinesWeightList.begin()
		//	  , 	  cur_iter_FIR_out= output_FIR_test_value.begin()
			  ;

		  for (;
		  cur_iter_stir_out!=cplus.end() 
			//  &&  
		//	  cur_iter_FIR_out!=output_FIR_test_value.end()
		;
		  ++cur_iter_stir_out  //++cur_iter_FIR_out
			  )
			  cerr << *cur_iter_stir_out << endl ;
/*		  	  check_if_equal(*cur_iter_stir_out, *cur_iter_FIR_out,
				  "check FIR filter implementation");    		  
				  */	   
	  }
		  		  
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 1)
  {
    cerr << "Usage : " << argv[0] << " \n";
    return EXIT_FAILURE;
  }
  BSplines_weightTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
