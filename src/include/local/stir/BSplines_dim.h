// 1
template <typename T, int num_dimensions>
inline 
T 
BSplines_dim(const Array<num_dimensions, T>& coeffs,
	 const BasicCoordinate<num_dimensions,pos_type>& relative_positions,
	 const BSplineType spline_type = cubic)
{
  T BSplines_value=0;

  	for (int k=int_pos-2; k<int_pos+3; ++k)		
	{	
		int index;
		if (k<0) index=-k;
		else if (k>=input_size) index=2*input_size-2-k;
		else index = k;
		assert(0<=index && index<input_size);
		BSplines_value += 
		  BSplines_dim_weight(k-relative_positions[1], spline_type)*
		  BSplines_dim(coeffs[index], cut_first_dimension(relative_positions));	

	}
}


#if defined( _MSC_VER) &&  _MSC_VER<=1300)
#define T float
#else
template <typename T>
#endif
inline 
T 
BSplines_dim(const Array<1, T>& coeffs,
	 const BasicCoordinate<1,pos_type>& relative_positions)
{
  T BSplines_value=0;

  for (int k=int_pos-2; k<int_pos+3; ++k)		
    {	
      int index;
      if (k<0) index=-k;
      else if (k>=input_size) index=2*input_size-2-k;
      else index = k;
      assert(0<=index && index<input_size);
      BSplines_value += 
	BSplines_dim_weight(k-relative_positions[1])*
	coeffs[index];      
    }
}
#if defined( _MSC_VER) &&  _MSC_VER<=1300)
#undef T
#endif

template <int num_dimensions, typename elemT>
void
  set_coef(Array<num_dimensions, elemT>& coefs, const Array<num_dimensions, elemT>& input,
	   z1,z2,lambda)
  {	
	
    Array<num_dimensions,elemT> temp ( input.get_index_range());
    BSplines_dim_coef(temp.begin(),temp.end(), 
		  input.begin(), input.end(), z1, z2, lamda);

    for (int i=coefs.get_min_index(); i<=coefs.get_max_index(); ++i)
      {
	set_coef(coef[i],
		 temp[i], 
		 z1, z2, lamda);
      }
  }

// 1d specialisation
template <typename elemT>
void
  set_coef(Array<1, elemT>& coefs, const Array<1, elemT>& input,
	   z1,z2,lambda)
  {	
	
    BSplines_dim_coef(coefs.begin(),coefs.end(), 
		  input.begin(), input.end(), z1, z2, lamda);
  }



#if 0
// TODO later
// fancy stuff to avoid having to repeat above convolution formulas for different kernels
/*
  value of convolution of an array with a separable kernel (given by f in all dimensions)

  get_value would be used normally to say get_value(coeffs,index) == coeffs[index],
  but allows to use e.g. periodic boundary conditions or extrapolation
*/

template <typename T, typename FunctionT, int num_dimensions>
inline 
T 
convolution(const Array<num_dimensions, T>& coeffs,
	    IndexingMethod get_value,
	    const BasicCoordinate<num_dimensions,pos_type>& relative_positions,
	    FunctionT f)
{
  T BSplines_value=0;

  	for (int k=int_pos-2; k<int_pos+3; ++k)		
	{	
		BSplines_value += 
		  f(k-relative_positions[1])*
		  convolution(get_value(coeffs,index), cut_first_dimension(relative_positions), f);	

	}
}


// 1d specialisation
template <typename T, typename FunctionT>
inline 
T 
convolution(const Array<1, T>& coeffs,
	 const BasicCoordinate<1,pos_type>& relative_positions,
	 FunctionT f)
{
  T BSplines_value=0;

  for (int k=int_pos-2; k<int_pos+3; ++k)		
    {	
      int index;
      if (k<0) index=-k;
      else if (k>=input_size) index=2*input_size-2-k;
      else index = k;
      assert(0<=index && index<input_size);
      BSplines_value += 
	f(k-relative_positions[1])*
	get_value(coeffs,index);
      
    }
}


template <typename T, int num_dimensions>
inline 
T 
BSplines_dim(const Array<num_dimensions, T>& coeffs,
	 const BasicCoordinate<num_dimensions,pos_type>& relative_positions)
{
  convolution(coeffs, relative_positions, BSplines_weight<T>);
}

template <typename T, int num_dimensions>
inline 
T 
BSplines_dim(const Array<num_dimensions, T>& coeffs,
	 const BasicCoordinate<num_dimensions,pos_type>& relative_positions)

#endif
