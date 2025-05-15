#ifndef NDEBUGXXX
{
  {
    const VoxelsOnCartesianGrid<float>& image = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*activity_image_sptr);
    const CartesianCoordinate3D<float> voxel_size = image.get_grid_spacing();
    CartesianCoordinate3D<float> origin = image.get_origin();
    const float z_to_middle = (image.get_max_index() + image.get_min_index()) * voxel_size.z() / 2.F;
    origin.z() -= z_to_middle;
    /* TODO replace with image.get_index_coordinates_for_physical_coordinates */
    info(format("first/last z for activity image after shift: {}/{}",
                (origin.z() + image.get_min_index() * voxel_size.z()),
                (origin.z() + image.get_max_index() * voxel_size.z())));
  }
  {
    const VoxelsOnCartesianGrid<float>& image = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*density_image_sptr);
    const CartesianCoordinate3D<float> voxel_size = image.get_grid_spacing();
    CartesianCoordinate3D<float> origin = image.get_origin();
    const float z_to_middle = (image.get_max_index() + image.get_min_index()) * voxel_size.z() / 2.F;
    origin.z() -= z_to_middle;
    /* TODO replace with image.get_index_coordinates_for_physical_coordinates */
    info(format("first/last z for attenuation image after shift: {}/{}",
                origin.z() + image.get_min_index() * voxel_size.z(),
                origin.z() + image.get_max_index() * voxel_size.z()));
  }
  {
    const VoxelsOnCartesianGrid<float>& image
        = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*get_density_image_for_scatter_points_sptr());
    const CartesianCoordinate3D<float> voxel_size = image.get_grid_spacing();
    CartesianCoordinate3D<float> origin = image.get_origin();
    const float z_to_middle = (image.get_max_index() + image.get_min_index()) * voxel_size.z() / 2.F;
    origin.z() -= z_to_middle;
    /* TODO replace with image.get_index_coordinates_for_physical_coordinates */
    info(format("first/last z for scatter-point image after shift: {}/{}",
                (origin.z() + image.get_min_index() * voxel_size.z()),
                (origin.z() + image.get_max_index() * voxel_size.z())));
  }
  {
    unsigned det_num_A, det_num_B;
    find_detectors(det_num_A, det_num_B, Bin(0, 0, this->proj_data_info_cyl_noarc_cor_sptr->get_min_axial_pos_num(0), 0));
    const float first = detection_points_vector[det_num_A].z();
    find_detectors(det_num_A, det_num_B, Bin(0, 0, this->proj_data_info_cyl_noarc_cor_sptr->get_max_axial_pos_num(0), 0));
    const float last = detection_points_vector[det_num_A].z();
    info(format("first/last z for detectors after shift: {}/{}", first, last));
  }
}
#endif

- sample

#ifndef NDEBUG
{
  const CartesianCoordinate3D<float> first = voxel_size * convert_int_to_float(min_index) + origin;
  const CartesianCoordinate3D<float> last = voxel_size * convert_int_to_float(max_index) + origin;
  info(format("Coordinates of centre of first and last voxel of scatter-point image after shifting to centre of scanner: "
              "{} / {} centre {}",
              first,
              last,
              ((first + last) / 2)));
}
#endif
