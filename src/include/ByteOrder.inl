//
// $Id$: $Date$
//

// for swap()
#include <algorithm>

inline ByteOrder::Order ByteOrder::get_native_order()
{ 
  // for efficiency, this refers to a static member.
  return native_order;
}


ByteOrder::ByteOrder(Order byte_order)
  : byte_order(byte_order)
{}

bool ByteOrder::operator==(ByteOrder order2) const
{
  return byte_order == order2.byte_order;
}

inline bool ByteOrder::is_native_order() const
{
  return (byte_order == native ||
    byte_order == get_native_order());
}

template <class NUMBER>
void ByteOrder::swap_if_necessary(NUMBER& a) const
{
  if (!is_native_order)
    ByteOrder::swap_order(a);
}



