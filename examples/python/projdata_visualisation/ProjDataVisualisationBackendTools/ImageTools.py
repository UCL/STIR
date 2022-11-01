import numpy as np


def float32_to_uint8(array: np.ndarray) -> np.ndarray:
    """
    This function converts a float32 array to an uint8 array.
    """
    array = array.astype(np.float32)
    if array.max() == array.min():
        # Avoid division by zero
        array = array.fill(1)
    else:
        # Normalize the image between int(0) and int(255)
        array = (array - array.min()) / (array.max() - array.min())
        array = array * 255

    array = array.astype(np.uint8)
    return array
