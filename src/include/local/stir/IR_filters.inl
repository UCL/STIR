using namespace std;

START_NAMESPACE_STIR

template <class BiIterType1,
		  class BiIterType2,
	      class BiIterType3,
		  class BiIterType4>
void 
inline 
IIR_filter(BiIterType1 output_begin_iterator, 
		   BiIterType1 output_end_iterator,
		   const BiIterType2 input_begin_iterator, 
		   const BiIterType2 input_end_iterator,
		   const BiIterType3 input_factor_begin_iterator,
		   const BiIterType3 input_factor_end_iterator,
		   const BiIterType4 pole_begin_iterator,
		   const BiIterType4 pole_end_iterator,
		   const bool if_initial_exists)
{
	// The input should be initialised to 0
//	if(output_begin_iterator==output_end_iterator)
//		warning("No output signal is given./n");

	if(if_initial_exists==false) 
	*output_begin_iterator=*input_factor_begin_iterator*(*input_begin_iterator);

	BiIterType1 current_output_iterator = output_begin_iterator ;
	BiIterType2 current_input_iterator = input_begin_iterator ;
	
	for(++current_output_iterator, ++current_input_iterator; 
	     current_output_iterator != output_end_iterator &&
		 current_input_iterator != input_end_iterator;
		 ++current_output_iterator,	++current_input_iterator)
		{
			BiIterType2 current_current_input_iterator = current_input_iterator;
			for(BiIterType3 current_input_factor_iterator = input_factor_begin_iterator;				
				current_input_factor_iterator != input_factor_end_iterator;
				++current_input_factor_iterator,--current_current_input_iterator
				)
				{					
					(*current_output_iterator) += (*current_input_factor_iterator)*
												   (*current_current_input_iterator);
				if (current_current_input_iterator==input_begin_iterator)
					break;
				}

			BiIterType4 current_pole_iterator = pole_begin_iterator;
			BiIterType1 current_feedback_iterator = current_output_iterator ;
			
			for(--current_feedback_iterator ;
				current_pole_iterator != pole_end_iterator 			
					;++current_pole_iterator,--current_feedback_iterator)
				{							
					(*current_output_iterator) -= (*current_pole_iterator)*
													(*current_feedback_iterator);			
					if(current_feedback_iterator==output_begin_iterator)
					break;
				}					
		}
}

template <class BiIterType1,
		  class BiIterType2,
	      class BiIterType3>		 
void 
inline 
FIR_filter(BiIterType1 output_begin_iterator, 
		   BiIterType1 output_end_iterator,
		   const BiIterType2 input_begin_iterator, 
		   const BiIterType2 input_end_iterator,
		   const BiIterType3 input_factor_begin_iterator,
		   const BiIterType3 input_factor_end_iterator,
		   const bool if_initial_exists)
{
	IIR_filter(output_begin_iterator, output_end_iterator, 
		   input_begin_iterator, input_end_iterator, 
		   input_factor_begin_iterator, input_factor_end_iterator,
		   output_begin_iterator, output_begin_iterator,if_initial_exists);
}
template <class BiIterType>
float
inline
sequence_sum(BiIterType input_iterator,
	   unsigned int Nmax,
	   float pole, // to be complex as well?
	   double precision 
	   )
{
	const int Nmax_precision = 4;//(int)ceil(log(precision)/log(abs(pole)));
	float sum=*input_iterator;
	for(int i=1; 
			i!=Nmax && i<=Nmax_precision //&& current_input_iterator!=end 
				; 							 ++i)
	  	  sum += *(++input_iterator)*pow(pole,i) ;
	return sum;
}   

template <class BiIterType1, 
		  class BiIterType2>
void
inline //temporarily
Bsplines_coef(BiIterType1 c_begin_iterator, 
			   BiIterType1 c_end_iterator,
			   BiIterType2 input_signal_begin_iterator, 
			   BiIterType2 input_signal_end_iterator)
{

#ifdef _MSC_VER<=1300

	typedef float c_value_type;
	typedef float input_value_type;
	//typedef typename _Val_Type(c_begin_iterator) c_value_type;
	//typedef typename _Val_Type(c_begin_iterator) input_value_type;
#else
	typedef typename std::iterator_traits<BiIterType1>::value_type c_value_type;
	typedef typename std::iterator_traits<BiIterType1>::value_type input_value_type;
#endif
	  	
	std::list<c_value_type> cplus, cminus, input_factor_for_cminus, pole_for_cplus, pole_for_cminus;
	std::list<input_value_type> input_factor_for_cplus;
  
	float z1 = -2. + sqrt(3.);
	pole_for_cplus.push_back(-z1);
	pole_for_cminus.push_back(-z1);
	input_factor_for_cplus.push_back(1);
	input_factor_for_cminus.push_back(-z1);
		
	for(BiIterType1 current_iterator=c_begin_iterator; 
	    current_iterator!=c_end_iterator; 
		++current_iterator)
		{
			cplus.push_back(0);
			cminus.push_back(0);
		}
	
	const int k=2*cplus.size()-3;
    *(cplus.begin())=sequence_sum(input_signal_begin_iterator,k,z1,.0001)/(1-pow(z1,k+1)) ; //k or Nmax_precision
	
	IIR_filter(cplus.begin(), cplus.end(),
		input_signal_begin_iterator, input_signal_end_iterator,
		input_factor_for_cplus.begin(), input_factor_for_cplus.end(),
		pole_for_cplus.begin(), pole_for_cplus.end(), 1);

	*--cminus.end() = z1*(*--cplus.end() + z1*(*--(--cplus.end())))/(1-z1*z1);
	IIR_filter(cminus.rbegin(), cminus.rend(),
		cplus.rbegin(), cplus.rend(),
		input_factor_for_cminus.begin(), input_factor_for_cminus.end(),
		pole_for_cminus.begin(), pole_for_cminus.end(), 1);
	
	for(BiIterType1 current_iterator=c_begin_iterator, current_cminus_iterator=cminus.begin(); 
	    current_iterator!=c_end_iterator &&	current_iterator!=cminus.end(); 
		++current_iterator,++current_cminus_iterator)
		{
			*current_iterator=*current_cminus_iterator*6;			
		}

}

/* recursively
template <class BiIterType>
float
inline
sequence_sum(BiIterType input_iterator,
	   unsigned int Nmax,
	   float pole // to be complex as well
	   )
{
	  static float sum=0;
	  if (Nmax=!0)
	  {
		  sum = *input_iterator*pole ;
		  sum += sequence_sum(++input_iterator,--Nmax, pole);                                                      		  
		  return sum;
	  }
	  else
		  return sum;
}   
*/

END_NAMESPACE_STIR
