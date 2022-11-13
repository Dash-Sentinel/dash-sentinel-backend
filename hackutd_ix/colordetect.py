import colorsys
from haishoku.haishoku import Haishoku

def get_image_color(filename):
    #dominant = Haishoku.getDominant(filename)
    dominant = Haishoku.getPalette(filename)[2][1]
    print(dominant)
    dominant = tuple(elem / 255 for elem in dominant)
    print(dominant)
    hls = colorsys.rgb_to_hls(*dominant)
    print(hls)
    hue = 360 * hls[0] # convert to degrees on a color wheel to make it easier to consult a chart
    # Determine color
    if hls[1] >= 0.8: # A very high lightness indicates white
        return "white"
    elif hls[1] <= 0.2: # A very low lightness indicates black
        return "black"
    elif hls[2] <= 0.1: # A very low saturation (if not white or black) indicates gray
        return "gray"
    else:
        match hue:
            case hue if hue <= 30:
                return "red"
            case hue if hue <= 90:
                return "yellow"
            case hue if hue <= 150:
                return "green"
            case hue if hue <= 210:
                return "cyan"
            case hue if hue <= 270:
                return "blue"
            case _:
                return "red"

## Tester code, remove later
import sys
print(get_image_color(sys.argv[1]))