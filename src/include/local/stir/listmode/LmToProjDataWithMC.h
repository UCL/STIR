
#ifndef __stir_listmode_LmToProjDataWithMC_H__
#define __stir_listmode_LmToProjDataWithMC_H__
//

#include "stir/recon_buildblock/BinNormalisation.h"
#include "local/stir/listmode/LmToProjData.h"
#include "stir/CartesianCoordinate3D.h"

#include "local/stir/motion/Polaris_MT_File.h"
#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"

START_NAMESPACE_STIR

class LmToProjDataWithMC : public LmToProjData
{
public:
     
  LmToProjDataWithMC(const char * const par_filename);

  virtual void get_bin_from_record(Bin& bin, const CListRecord& record, 
				   const double time, 
				   const ProjDataInfoCylindrical&) const;

  shared_ptr<BinNormalisation> normalisation_ptr;
  //shared_ptr<Polaris_MT_File> mt_file_ptr;

private:

  void find_ref_pos_from_att_file (float& att_start_time, float& att_end_time, 
	const float transmission_duration, const string attnuation_filename);
  
  void find_cartesian_coordinates_given_scanner_coordinates (CartesianCoordinate3D<float>& coord_1,
				 CartesianCoordinate3D<float>& coord_2,
				 const int Ring_A,const int Ring_B, 
				 const int det1, const int det2, 
				 const Scanner& scanner) const;


  Succeeded find_scanner_coordinates_given_cartesian_coordinates(int& det1, int& det2, int& ring1, int& ring2,
							  const CartesianCoordinate3D<float>& coord_1_in,
							  const CartesianCoordinate3D<float>& coord_2_in, 
							  const Scanner& scanner) const;

  void transform_detector_pair_into_view_bin (int& view,int& bin, 
					    const int det1,const int det2, 
					    const Scanner& scanner) const;

  shared_ptr<ProjDataInfo> proj_data_info_cyl_uncompressed_ptr;
  shared_ptr<Scanner> scanner_ptr;
  shared_ptr<RigidObject3DMotion> ro3d_ptr;
  //TODO somehow get this from RigidObject3DMotionFromPolaris
  string mt_filename;
  string attenuation_filename; 
  string norm_filename;  
  float transmission_duration;
   
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  mutable RigidObject3DTransformation ro3d_move_to_reference_position;

  
};

END_NAMESPACE_STIR


#endif
