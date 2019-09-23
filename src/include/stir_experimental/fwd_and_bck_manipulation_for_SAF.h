
#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/VectorWithOffset.h"
#include "stir/SegmentByView.h"
#include "stir_experimental/recon_buildblock/ProjMatrixByDensel.h"



START_NAMESPACE_STIR
void
fwd_project(ProjData& proj_data,VoxelsOnCartesianGrid<float>* vox_image_ptr,
		const int start_segment_num, const int end_segment_num,
		const int start_axial_pos_num, const int end_axial_pos_num,		
		const int start_view, const int end_view,
		const int start_tang_pos_num,const int end_tang_pos_num);



void
do_segments_densels_fwd(const VoxelsOnCartesianGrid<float>& image, 
            ProjData& proj_data,
	    VectorWithOffset<SegmentByView<float> *>& all_segments,
            const int min_z, const int max_z,
            const int min_y, const int max_y,
            const int min_x, const int max_x,
	    ProjMatrixByDensel& proj_matrix);


void
fwd_densels_all(VectorWithOffset<SegmentByView<float> *>& all_segments,
		shared_ptr<ProjMatrixByDensel> proj_matrix_ptr, 
		shared_ptr<ProjData > proj_data_ptr,
		const int min_z, const int max_z,
		const int min_y, const int max_y,
		const int min_x, const int max_x,
		const DiscretisedDensity<3,float>& in_density);

void
find_inverse_and_bck_densels(DiscretisedDensity<3,float>& image,
			     VectorWithOffset<SegmentByView<float> *>& all_segments,
			     VectorWithOffset<SegmentByView<float> *>& attenuation_segmnets,			
				const int min_z, const int max_z,
				const int min_y, const int max_y,
				const int min_x, const int max_x,
				ProjMatrixByDensel& proj_matrix, 
				bool do_attenuation,
				const float threshold, bool normalize_result);




END_NAMESPACE_STIR
