//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \ingroup numerics_buildblock
  \brief Implementation of the (cubic) B-Splines Interpolation 

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/

using namespace std;

START_NAMESPACE_STIR

template <class T>
static inline 
void set_to_zero(T& v)
{
	v=0;
}

template <class T, int num_dimensions>
static inline 
void set_to_zero(Array<num_dimensions,T>& v)
{
	v.fill(0);
}

template <class T>
static inline 
void set_to_zero(std::vector<T>& v)
{
	for (typename std::vector<T>::iterator iter = v.begin(); 
		iter != v.end(); ++iter)
		set_to_zero(*iter);
}

template <typename out_elemT, typename in_elemT>
BSplines1DRegularGrid<out_elemT,in_elemT>::
BSplines1DRegularGrid()
{ }

template <typename out_elemT, typename in_elemT>
BSplines1DRegularGrid<out_elemT,in_elemT>::
 BSplines1DRegularGrid(const std::vector<in_elemT> & input_vector)
{ 
	BSplines1DRegularGrid<out_elemT, in_elemT>::
		set_coef(input_vector.begin(), input_vector.end());	
}

/*
template <typename out_elemT, typename in_elemT>
BSplines1DRegularGrid<out_elemT,in_elemT>::
BSplines1DRegularGrid(const RandIterIn input_begin_iterator, //explicit??
				  const RandIterIn input_end_iterator)
{
	BSplines1DRegularGrid<out_elemT, in_elemT>::
		set_coef(input_begin_iterator, input_end_iterator);
}
*/

template <typename out_elemT, typename in_elemT>
BSplines1DRegularGrid<out_elemT,in_elemT>::
~BSplines1DRegularGrid()
{}

template <class IterT>
#if defined(_MSC_VER) && _MSC_VER<=1300
  float
#else
typename  std::iterator_traits<IterT>::value_type
#endif
cplus0(const IterT input_begin_iterator,
	   const IterT input_end_iterator,
	   unsigned int Nmax,  double pole, // to be complex as well?
	   const double precision, const bool periodicity)
{
#if defined(_MSC_VER) && _MSC_VER<=1300
  typedef float out_elemT;
#else
  typedef typename std::iterator_traits<IterT>::value_type out_elemT;
#endif
	
	const int input_size = input_end_iterator - input_begin_iterator;
	const int Nmax_precision = 
		(periodicity==0? 2*input_size-2 : input_size);//(int)ceil(log(precision)/log(abs(pole)));
//	std::cerr << " prec " << log(precision)/log(abs(pole));
	out_elemT sum=*input_begin_iterator;
	for(int i=1; 
		i<Nmax_precision;
		++i)
	{
		int index = i;
		if (periodicity==0 && i >= input_size)
			index = 2*input_size-2-i;
	  	  sum += *(input_begin_iterator+index)*pow(pole,i) ;
	}
			//cerr << "SUM:" << sum << endl ;
	return	sum;
}   

template <class RandIterOut, class IterT>
void
BSplines_coef(RandIterOut c_begin_iterator, 
			   RandIterOut c_end_iterator,
			   IterT input_begin_iterator, 
			   IterT input_end_iterator)
{

#if defined(_MSC_VER) && _MSC_VER<=1300
	typedef float out_elemT;
	typedef float in_elemT;
	//typedef typename _Val_Type(c_begin_iterator) out_elemT;
	//typedef typename _Val_Type(c_begin_iterator) in_elemT;
#else
	typedef typename std::iterator_traits<RandIterOut>::value_type out_elemT;
	typedef typename std::iterator_traits<RandIterOut>::value_type in_elemT;
#endif
	  	
/*		
		cplus(c_end_iterator-c_begin_iterator, 0)
		cminus(c_end_iterator-c_begin_iterator, 0),
*/
	float z1 = -2. + sqrt(3.);
	const int input_size = c_end_iterator-c_begin_iterator ;//-1; //!!!

	typedef std::vector<out_elemT> c_vector_type;
	c_vector_type cplus(c_end_iterator-c_begin_iterator), 
		cminus(c_end_iterator-c_begin_iterator);
	std::vector<double>
		input_factor_for_cminus(1, -z1), pole_for_cplus(1, -z1), pole_for_cminus(1,-z1);
	set_to_zero(cplus);
	set_to_zero(cminus);
	std::vector<double> input_factor_for_cplus(1, (double)1);
		
	const int k=2*input_size-3;
	assert(k>=0);
    *(cplus.begin())=cplus0(
		input_begin_iterator,input_end_iterator, k,z1,.00001,0)/(1-pow(z1,k+1)) ; //k or Nmax_precision
	
	IIR_filter(cplus.begin(), cplus.end(),
		input_begin_iterator, input_end_iterator,
		input_factor_for_cplus.begin(), input_factor_for_cplus.end(),
		pole_for_cplus.begin(), pole_for_cplus.end(), 1);

	*(cminus.end()-1) = (*(cplus.end()-1) + (*(cplus.end()-2))*z1)*z1/(z1*z1-1);
	IIR_filter(cminus.rbegin(), cminus.rend(),
		cplus.rbegin(), cplus.rend(),
		input_factor_for_cminus.begin(), input_factor_for_cminus.end(),
		pole_for_cminus.begin(), pole_for_cminus.end(), 1);
	
	RandIterOut current_iterator=c_begin_iterator;
	typename c_vector_type::const_iterator current_cminus_iterator=cminus.begin();
	for(;
         current_iterator!=c_end_iterator &&	current_cminus_iterator!=cminus.end(); 
		++current_iterator,++current_cminus_iterator)
		{
			*current_iterator=*current_cminus_iterator*6;			
		}	
}

template <typename pos_type>
pos_type 
BSplines_weight(const pos_type abs_relative_position
					//, enum enum_spline_level
					) 
{
	if (abs_relative_position>=2)
		return 0;
	if (abs_relative_position>=1)		
		return pow((2-abs_relative_position),3)/6;
	if (abs_relative_position>=0)		
		return 2./3. + (0.5*abs_relative_position-1)*abs_relative_position*abs_relative_position;
	else
	//	warning("BSplineWeight does not take negative values as input!");
		return EXIT_FAILURE;
//	-100;
}

template <typename pos_type>
pos_type 
BSplines_1st_der_weight(const pos_type abs_relative_position) 
{
	if (abs_relative_position>=2)
		return 0;
	if (abs_relative_position>=1)		
		return -0.5*(abs_relative_position-2)*(abs_relative_position-2);
	if (abs_relative_position>=0)		
		return abs_relative_position*(1.5*abs_relative_position-2.);
	else
	//	warning("BSplines_1st_der_weight does not take negative values as input!");
		return EXIT_FAILURE;
//	-100;
}

#if 0
// needs to be in .h for VC 6.0
template <typename out_elemT, typename in_elemT>
void
  BSplines1DRegularGrid<out_elemT,in_elemT>::
  set_coef(RandIterIn input_begin_iterator, RandIterIn input_end_iterator)
  {	
	input_size = input_end_iterator - input_begin_iterator;
	for(int i=1; i<=input_size; ++i)
		BSplines_coef_vector.push_back(-1); // Giving one more gives correct result for the last value // Now I give 2 at the end ?
	
			BSplines_coef(BSplines_coef_vector.begin(),BSplines_coef_vector.end(), 
			input_begin_iterator, input_end_iterator);		
		//assert (input_size==static_cast<int>(BSplines_coef_vector.size()-2));
  }
#endif

template <typename out_elemT, typename in_elemT>
out_elemT 
BSplines1DRegularGrid<out_elemT,in_elemT>::
BSpline(const pos_type relative_position) 
{	
	out_elemT BSpline_value;
	set_to_zero(BSpline_value);
	//assert(relative_position<static_cast<out_elemT>(input_size));
//		warning("New sampling position out of range");
	for (int k=(int)relative_position-2; k<(int)(relative_position+3) && k<=input_size; ++k)		
	{
		if (k<0) continue;
		BSpline_value += 
			BSplines1DRegularGrid<out_elemT,in_elemT>::BSplines_coef_vector[k] *
			BSplines_weight(fabs((pos_type)k-relative_position));
	}
	return BSpline_value;
}

template <typename out_elemT, typename in_elemT>
out_elemT 
BSplines1DRegularGrid<out_elemT,in_elemT>::
BSpline_1st_der(const pos_type relative_position) 
{	
//	assert(relative_position<static_cast<pos_type>(input_size));
	out_elemT BSpline_value;
	set_to_zero(BSpline_value);
	for (int k=(int)relative_position-2; k<(int)(relative_position+3) && k<=input_size; ++k)		
	{
		if (k<0) continue;
		BSpline_value += 
			BSplines1DRegularGrid<out_elemT,in_elemT>::BSplines_coef_vector[k] *
			BSplines_1st_der_weight(fabs((pos_type)k-relative_position));
	}
	return BSpline_value;
}

template <typename out_elemT, typename in_elemT>
const out_elemT BSplines1DRegularGrid<out_elemT,in_elemT>::
operator() (const pos_type relative_position) const 
{
	return BSplines1DRegularGrid<out_elemT,in_elemT>::
		BSpline(relative_position);		
};
template <typename out_elemT, typename in_elemT>
out_elemT BSplines1DRegularGrid<out_elemT,in_elemT>::
operator() (const pos_type relative_position)
{
	return BSplines1DRegularGrid<out_elemT,in_elemT>::
		BSpline(relative_position);		
}
//*
template <typename out_elemT, typename in_elemT>
const std::vector<out_elemT> BSplines1DRegularGrid<out_elemT,in_elemT>::
BSpline_output_sequence(RandIterOut output_relative_position_begin_iterator,  //relative_position might be better float
						RandIterOut output_relative_position_end_iterator)
{
	std::vector<pos_type> output_vector(output_relative_position_end_iterator-
		output_relative_position_begin_iterator);	
	
	for(RandIterOut current_iterator=output_vector.begin(), 
		 current_relative_position_iterator=output_relative_position_begin_iterator; 
	    current_iterator!=output_vector.end() && 
			current_relative_position_iterator!=output_relative_position_end_iterator; 
		++current_iterator,++current_relative_position_iterator)
			*current_iterator = BSplines1DRegularGrid<out_elemT,in_elemT>:: 
			BSpline(*current_relative_position_iterator);		

	return output_vector;		
}
template <typename out_elemT, typename in_elemT>
const std::vector<out_elemT> BSplines1DRegularGrid<out_elemT,in_elemT>::
BSpline_output_sequence(std::vector<pos_type> output_relative_position)
{
	return BSpline_output_sequence(output_relative_position.begin(),
						output_relative_position.end());
}


//*/

END_NAMESPACE_STIR
