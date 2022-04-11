import math
import numpy as np
def get_color(x, color_maps):
    x1 = math.floor(x)
    x2 = math.ceil(x)
    t = x - x1

    return rgb_to_hex(t * color_maps[x1] + (1-t) * color_maps[x2])

def rgb_to_hex(rgb):
    rgb = tuple([int(i) for i in rgb])
    return '#%02x%02x%02x' % rgb



if __name__ == "__main__":
    colors = np.array([(233,255,242), (186,255,216), (135,255,187), (73,255,152), (0,255,127)])
    print(get_color(0, colors))