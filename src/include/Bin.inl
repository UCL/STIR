//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations of inline functions of class Bin

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$
*/

START_NAMESPACE_TOMO

Bin::Bin()
{}


Bin::Bin(int segment_num,int view_num, int axial_pos_num,int tangential_pos_num,float bin_value)
	 :segment(segment_num),view(view_num),
	 axial_pos(axial_pos_num),tangential_pos(tangential_pos_num),bin_value(bin_value)
     {}

     
int
 Bin:: axial_pos_num()const
{ return axial_pos;}

int
 Bin::segment_num()const 
{return segment;}

int
 Bin::tangential_pos_num()  const
{ return tangential_pos;}

int
 Bin::view_num() const
{ return view;}

int&
 Bin::axial_pos_num()
{ return axial_pos;}

int&
 Bin:: segment_num()
{return segment;}

int&
 Bin::tangential_pos_num()
{ return tangential_pos;}

int&
 Bin:: view_num() 
{ return view;}

#if 0
const ProjDataInfo *
Bin::get_proj_data_info_ptr() const
{
  return proj_data_info_ptr.get();
}
#endif

Bin
Bin::get_empty_copy() const
{
 
  Bin copy(segment_num(),view_num(),axial_pos_num(),tangential_pos_num(),0);

  return copy;
}

float 
Bin::get_bin_value()const 
{ return bin_value;}

void
Bin::set_bin_value( float v )
{ bin_value = v ;}

Bin&  
Bin::operator+=(const float dx) 
{ bin_value+=dx;  
  return *this;}



END_NAMESPACE_TOMO
