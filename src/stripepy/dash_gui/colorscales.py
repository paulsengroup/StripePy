from stripepy.plot import _get_custom_palettes

custom_palettes = _get_custom_palettes()

fruit_punch = custom_palettes["fruit_punch"] * 255
fruit_punch = tuple(map(tuple, fruit_punch.tolist()))  # ((255, 255, 255), (255, 204, 204), (255, 153, 153), ...)

fall = custom_palettes["fall"] * 255
fall = tuple(map(tuple, fall.tolist()))

blues = custom_palettes["blues"] * 255
blues = tuple(map(tuple, blues.tolist()))

acidblues = custom_palettes["acidblues"] * 255
acidblues = tuple(map(tuple, acidblues.tolist()))

nmeth = custom_palettes["nmeth"] * 255
nmeth = tuple(map(tuple, nmeth.tolist()))


def color_scale(name):
    """
    Provide colorscales to interface.py.
    Utilizes 255-based RGB-scales adapted from https://github.com/open2c/cooltools/blob/master/cooltools/lib/plotting.py

    Returns
    -------
    List of tuples. Each tuple consists of a floating number and an rgb string on the form 'rgb(255, 255, 255)'.
    """
    if name is None:
        name = "fruit_punch"
    scales = {
        "fruit_punch": [
            [index / (len(fruit_punch) - 1), "rgb" + str(element)] for index, element in enumerate(fruit_punch)
        ],
        "fall": [[index / (len(fall) - 1), "rgb" + str(element)] for index, element in enumerate(fall)],
        "blues": [[index / (len(blues) - 1), "rgb" + str(element)] for index, element in enumerate(blues)],
        "acidblues": [[index / (len(acidblues) - 1), "rgb" + str(element)] for index, element in enumerate(acidblues)],
        "nmeth": [[index / (len(nmeth) - 1), "rgb" + str(element)] for index, element in enumerate(nmeth)],
    }
    return scales[name]


def contrast(colorMap, location):
    """
    Provide colors to hover labels, stripes and background. Recommended use is inside another function, like this: function_that_takes_color_name(contrast(colorMap, "stripe"))

    Returns
    -------
    String. Name of color to be inserted in the location of the call.
    """
    label = {
        "fruit_punch": "teal",
        "fall": "teal",
        "blues": "teal",
        "acidblues": "teal",
        "nmeth": "purple",
    }
    stripe = {
        "fruit_punch": "cornflowerblue",
        "fall": "deepskyblue",
        "blues": "magenta",
        "acidblues": "coral",
        "nmeth": "coral",
    }
    background = {
        "fruit_punch": "mediumslateblue",
        "fall": "mediumslateblue",
        "blues": "chartreuse",
        "acidblues": "yellow",
        "nmeth": "cyan",
    }
    if location == "label":
        return label[colorMap]
    elif location == "stripe":
        return stripe[colorMap]
    elif location == "background":
        return background[colorMap]
