//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock
  \brief Declaration of RayTraceVoxelsOnCartesianGrid

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$
*/
#include "Tomography_common.h"

START_NAMESPACE_TOMO

class ProjMatrixElemsForOneBin;
template <typename elemT> class CartesianCoordinate3D;

/*!
  \brief RayTraceVoxelsOnCartesianGrid finds the Length of Intersections (LOIs)
  of an LOR with a grid of voxels and appends them to
  the ProjMatrixElemsForOneBin object.

  \param lor object to which the intersected voxels and the LOI will be appended
  \param start_point first point on the LOR. The first voxel will contain this point.
  \param stop_point last point on the LOR. The last voxel will contain this point.
  \param voxel_size normally in mm
  \param normalisation_constant LOIs will be multiplied with this constant

  start_point and end_point have to be given in 'voxel grid
  units' (i.e. voxels are spaced 1 unit apart). The centre
  of the voxels are assumed to be at integer coordinates 
  (e.g. (0,0,0) is the centre of a voxel).

  \warning RayTraceVoxelsOnCartesianGrid appends voxels and intersection lengths to the lor.
  It does NOT reset it first.

  \warning The current implementation assumes that 
  \code
  start_point.x() >= stop_point.x()
  start_point.y() <= stop_point.y()
  start_point.z() <= stop_point.z()
  \endcode

  RayTraceVoxelsOnCartesianGrid uses Siddon's algorithm.

  Siddon's algorithm works by looking at intersections of the 
  'intra-voxel' planes with the LOR.

  The LORs is parametrised as
  \code
  (x,y,z) = a (1/inc_x, 1/inc_y, 1/inc_z) + start_point
  \endcode
  Then values of 'a' are computed where the LOR intersects an intra-voxel plane.
  For example, 'ax' are the values where x= n + 0.5 (for integer n).
  Finally, we can go along the LOR and check which of the ax,ay,az is smallest,
  as this determines which plane the LOR intersects at this point.

*/
void 
RayTraceVoxelsOnCartesianGrid(ProjMatrixElemsForOneBin& lor, 
                              const CartesianCoordinate3D<float>& start_point, 
                              const CartesianCoordinate3D<float>& end_point, 
                              const CartesianCoordinate3D<float>& voxel_size,
                              const float normalisation_constant = 1.F);

END_NAMESPACE_TOMO
