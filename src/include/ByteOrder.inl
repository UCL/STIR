//
// $Id$: $Date$
//
/*!
  \file 
 
  \brief This file declares the ByteOrder class.

  \author Kris Thielemans 
  \author Alexey Zverovich
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

START_NAMESPACE_TOMO

ByteOrder::ByteOrder(Order byte_order)
  : byte_order(byte_order)
{}

/*! for efficiency, this refers to the static member native_order.*/
inline ByteOrder::Order ByteOrder::get_native_order()
{ 
  return native_order;
}

inline bool ByteOrder::is_native_order() const
{
  return 
    byte_order == native ||
    byte_order == get_native_order();
}

/*! This takes care of interpreting 'native' and 'swapped'. */
bool ByteOrder::operator==(const ByteOrder order2) const
{
  // Simple comparison (byte_order == order2.byte_order)
  // does not work because of 4 possible values of the enum.
  return 
    (is_native_order() && order2.is_native_order()) ||
    (!is_native_order() && !order2.is_native_order());
}

bool ByteOrder::operator!=(const ByteOrder order2) const
{
return !(*this == order2);
}

END_NAMESPACE_TOMO
