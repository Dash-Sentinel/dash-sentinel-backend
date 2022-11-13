import colorsys
from haishoku.haishoku import Haishoku
from common import Color

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
        return Color.WHITE
    elif hls[1] <= 0.2: # A very low lightness indicates black
        return Color.BLACK
    elif hls[2] <= 0.1: # A very low saturation (if not white or black) indicates gray
        return Color.GRAY
    else:
        match hue:
            case hue if hue <= 30:
                return Color.RED
            case hue if hue <= 90:
                return Color.YELLOW
            case hue if hue <= 150:
                return Color.GREEN
            case hue if hue <= 210:
                return Color.CYAN
            case hue if hue <= 270:
                return Color.BLUE
            case _:
                return Color.RED

## Tester code, remove later
#import sys
#print(get_image_color(sys.argv[1]))