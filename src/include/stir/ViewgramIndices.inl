/*!
  \file
  \ingroup projdata

  \brief inline implementations for class stir::ViewgramIndices

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

ViewgramIndices::ViewgramIndices()
    : SegmentIndices(),
      _view(0)
{}

ViewgramIndices::ViewgramIndices(const int view_num, const int segment_num)
    : SegmentIndices(segment_num),
      _view(view_num)
{}

int
ViewgramIndices::view_num() const
{
  return _view;
}

int&
ViewgramIndices::view_num()
{
  return _view;
}

bool
ViewgramIndices::operator<(const ViewgramIndices& other) const
{
  return (_view < other._view) || ((_view == other._view) && base_type::operator<(other));
}

bool
ViewgramIndices::operator==(const ViewgramIndices& other) const
{
  return (_view == other._view) && base_type::operator==(other);
}

bool
ViewgramIndices::operator!=(const ViewgramIndices& other) const
{
  return !(*this == other);
}
END_NAMESPACE_STIR
