from plotly.express.colors import named_colorscales

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
    Choose between cooltools colors and default plotly colors.

    Returns
    -------
    List of tuples. Each tuple consists of a floating number and an rgb string on the form 'rgb(255, 255, 255)'.
    If no match between input string and stored values, falls back to default plotly colors.
    """
    if name is None:
        name = "fruit_punch "
    scales = {
        "fruit_punch ": [
            [index / (len(fruit_punch) - 1), "rgb" + str(element)] for index, element in enumerate(fruit_punch)
        ],
        "fall ": [[index / (len(fall) - 1), "rgb" + str(element)] for index, element in enumerate(fall)],
        "blues ": [[index / (len(blues) - 1), "rgb" + str(element)] for index, element in enumerate(blues)],
        "acidblues ": [[index / (len(acidblues) - 1), "rgb" + str(element)] for index, element in enumerate(acidblues)],
        "nmeth ": [[index / (len(nmeth) - 1), "rgb" + str(element)] for index, element in enumerate(nmeth)],
    }

    if name not in scales:
        return named_colorscales[name]
    return scales[name]
