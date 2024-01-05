/*!
  \file
  \ingroup projdata

  \brief inline implementations for class stir::SegmentIndices

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR

SegmentIndices::SegmentIndices(const int segment_num, const int timing_pos_num)
  : _segment(segment_num), _timing_pos(timing_pos_num)
{}

int
SegmentIndices::segment_num() const
{
  return _segment;
}

int&
SegmentIndices::segment_num()
{
  return _segment;
}

int
SegmentIndices::timing_pos_num() const
{
  return _timing_pos;
}

int&
SegmentIndices::timing_pos_num()
{
  return _timing_pos;
}

bool
SegmentIndices::operator<(const SegmentIndices& other) const
{
  return (_segment < other._segment) || ((_segment == other._segment) && (_timing_pos < other._timing_pos));
}

bool
SegmentIndices::operator==(const SegmentIndices& other) const
{
  return (_segment == other._segment) && (_timing_pos == other._timing_pos);
}

bool
SegmentIndices::operator!=(const SegmentIndices& other) const
{
  return !(*this == other);
}
END_NAMESPACE_STIR
