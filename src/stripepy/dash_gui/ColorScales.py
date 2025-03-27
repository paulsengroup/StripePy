import numpy as np

from stripepy.plot import _get_custom_palettes

custom_palettes = _get_custom_palettes()

fruit_punch = custom_palettes["fruit_punch"] * 255
fruit_punch = tuple(map(tuple, fruit_punch.tolist()))  # ((255, 255, 255), (255, 255, 204), ...)

fall = custom_palettes["fall"] * 255
fall = tuple(map(tuple, fall.tolist()))

blues = custom_palettes["blues"] * 255
blues = tuple(map(tuple, blues.tolist()))

acidblues = custom_palettes["acidblues"] * 255
acidblues = tuple(map(tuple, acidblues.tolist()))

nmeth = custom_palettes["nmeth"] * 255
nmeth = tuple(map(tuple, nmeth.tolist()))


def color_scale(name):
    if name is None:
        name = "fruit_punch"
    scales = {
        "fruit_punch": [
            [index / (len(fruit_punch) - 1), "rgb" + str(element)] for index, element in enumerate(fruit_punch)
        ],
        "fall": [[index / (len(fall) - 1), "rgb" + str(element)] for index, element in enumerate(fall)],
        "blues_new": [[index / (len(blues) - 1), "rgb" + str(element)] for index, element in enumerate(blues)],
        "acidblues": [[index / (len(acidblues) - 1), "rgb" + str(element)] for index, element in enumerate(acidblues)],
        "nmeth": [[index / (len(nmeth) - 1), "rgb" + str(element)] for index, element in enumerate(nmeth)],
    }

    return scales[name]
