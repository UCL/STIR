/*!
  \file
  \ingroup projdata

  \brief inline implementations for class stir::SinogramIndices

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/


START_NAMESPACE_STIR

SinogramIndices::SinogramIndices()
:SegmentIndices(),_axial_pos(0)
  {}

SinogramIndices::SinogramIndices( const int axial_pos_num,const int segment_num, const int timing_pos_num)
  : SegmentIndices(segment_num, timing_pos_num),_axial_pos(axial_pos_num)
  {}

SinogramIndices::SinogramIndices(const Bin& bin)
  : SegmentIndices(bin),_axial_pos(bin.axial_pos_num())
  {}

int 
SinogramIndices::axial_pos_num() const
{
  return _axial_pos;}


int& 
SinogramIndices::axial_pos_num() 
{ return _axial_pos;}

bool 
SinogramIndices::
operator<(const SinogramIndices& other) const
{
  return (_axial_pos< other._axial_pos) ||
    ((_axial_pos == other._axial_pos) && base_type::operator<(other));
}

bool 
SinogramIndices::
operator==(const SinogramIndices& other) const
{
  return (_axial_pos == other._axial_pos) && base_type::operator==(other);
}

bool 
SinogramIndices::
operator!=(const SinogramIndices& other) const
{
  return !(*this == other);
}
END_NAMESPACE_STIR
