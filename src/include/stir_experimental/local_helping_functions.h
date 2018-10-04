



#include "stir/Array.h"
#include "stir/VoxelsOnCartesianGrid.h"



/*  Here there are helping function used locally in ModifiedInverseAverigingImageFilter.cxx

*/


START_NAMESPACE_STIR

// Complex multiplication
void mulitply_complex_arrays(Array<1,float>& out_array, const Array<1,float>& array_nom,
			      const Array<1,float>& array_denom);

// Complex division 
void divide_complex_arrays( Array<1,float>& out_array, const Array<1,float>& array_nom,
			      const Array<1,float>& array_denom);

// two argument implementation

void mulitply_complex_arrays(Array<1,float>& array_nom,
			      const Array<1,float>& array_denom);
void divide_complex_arrays( Array<1,float>& array_nom,
			     const Array<1,float>& array_denom);

// convert 3D arra into 1D array
void convert_array_3D_into_1D_array ( Array<1,float>& out_array,const Array<3,float>& in_array);
// convert 1d array into 3d array 
void convert_array_1D_into_3D_array( Array<3,float>& out_array,const Array<1,float>& in_array);
// create 3d kernel
void create_kernel_3d ( Array<3,float>& kernel_3d, const VectorWithOffset < float>& kernel_1d);
void create_kernel_2d ( Array<2, float> & kernel_2d, const VectorWithOffset < float>& kernel_1d);
// padd filter coefficients and make them symmetric
void padd_filter_coefficients_3D_and_make_them_symmetric(VectorWithOffset < VectorWithOffset < VectorWithOffset < float> > > &padded_filter_coefficients_3D,
							 VectorWithOffset < VectorWithOffset < VectorWithOffset < float> > > &filter_coefficients);


void convert_array_2D_into_1D_array( Array<1,float>& out_array,Array<2,float>& in_array);

void convert_array_1D_into_2D_array( Array<2,float>& out_array,Array<1,float>& in_array);


void precompute_filter_coefficients_for_second_apporach(VoxelsOnCartesianGrid<float>& precomputed_coefficients,
							const VoxelsOnCartesianGrid<float>& input_image,
							VoxelsOnCartesianGrid<float>& sensitivity_image,
							VoxelsOnCartesianGrid<float>& normalised_bck);
	      




END_NAMESPACE_STIR
