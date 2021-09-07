from riemann.riemann_main import *

if __name__ == '__main__':
    sys.setrecursionlimit(3000)
    # foos = [lambda x: math.exp(x / 10) * math.sin(x) ** 2 + 1.2,
    #         lambda x: x * (x - 2) * (x + 2) * (x - 5) * (x + 5) * math.exp(-abs(x)) * math.sin(x) / 12 + 1,
    #         lambda x: (abs(x) - abs(x - 1) + x ** 2 / 15) / 2 + 1.2]
    # divs = [[(-5, -4, -3, -2, -2.5, 0, .5, 2, 3, 4.5, 5),
    #          (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
    #          (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
    #          (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)],
    #         [(-5, -4, -3, -2, -2.5, 0, .5, 2, 3, 4.5, 5),
    #          (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
    #          (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
    #          (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)][::-1],
    #         [(-5, -4, -3, -2, -2.5, 0, .5, 2, 3, 4.5, 5),
    #          (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
    #          (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
    #          (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)]]
    #
    # smooth_lower_sums_op(divs, foos, (1280, 720), .25)

    # seq = [ [-4, -4, -4, -4, -3.1, -2, -2, -1, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4],
    #         [-4, -3, -3, -2, -2, -1.3, -1.1, 0, .6, 1.8, 2.2, 2.6, 2.7, 2.8, 3, 3.3, 3.6, 4],
    #         [-4, -3.5, -3, -2, -1.7, -1.3, -1, -.3, .5, 1.5, 2, 2.2, 2.4, 2.6, 2.9, 3, 3.8, 4],
    #        [-4, -3.8, -3.5, -3.2, -3, -2.5, -2.2, -1.9, -1, .4, 1.2, 1.8, 2, 2.4, 2.8, 3, 3.2, 4]]
    #
    # triple_moving(divisions=seq, speed=.33, resolution=FHD)
    # absolute_value(lambda x: -x*(x-2)*(x+2)*(x-5)*(x+5)*math.exp(-abs(x))*math.sin(x)/12 + 1, speed=.1, resolution=FHD,
    #                filename='abs_fast.mp4', id_='abs_fast')
    # hill(.1, FHD, 'exp_hill.mp4')
    # seq = [[0, .1, .1, .2, .4, .4, .55, .63, .71, .85, 1],
    #        [0, .12, .18, .23, .36, .36, .5, .6, .7, .85, 1],
    #        [0, .05, .11, .2, .3, .38, .45, .64, .7, .9, 1],
    #        [0, .1, .1, .1, .3, .3, .3, .7, .7, .7, 1],
    #        [0, .08, .15, .24, .36, .45, .6, .75, .88, .95, 1],]
    #
    # smooth_animate_divisions(seq, FHD, .25, bigger_ab_at=1, radius_foo=2)
    # # smooth_animate_divisions(seq, FHD, .25, bigger_ab_at=1, radius_foo=3)
    # bold([-1, -.4, -.15, .1, .3, .45, .7, .9, 1], [.45, .7], lambda x: math.exp(x / 2) * math.sin(5 * x) ** 2 + 1.2,
    #      resolution=FHD, speed=.2, light_color='beige', dark_color='irresistible')
    foos = [lambda x: math.exp(x / 10) * math.sin(x) ** 2 + 1.2, lambda x: x*(x-2)*(x+2)*(x-5)*(x+5)*math.exp(-abs(x))*math.sin(x)/12+1,
            lambda x: (abs(x) - abs(x-1) + x**2/15)/2 + 1.2][:2]
    divs = [[(-5, -4, -3, -2, -1.5, 0, .5, 2, 3, 4.5, 5),
            (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
            (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
            (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)],
            [(-5, -4, -3, -2, -1.5, 0, .5, 2, 3, 4.5, 5),
             (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
             (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
             (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)][::-1],
            [(-5, -4, -3, -2, -1.5, 0, .5, 2, 3, 4.5, 5),
             (-5, -4.5, -3.5, -1.5, -.5, 1, 1.5, 2.5, 3.5, 4, 5),
             (-5, -4.3, -3, -2.2, -1, -.5, 1.2, 2.3, 3, 3.5, 5),
             (-5, -4.7, -4, -3.3, -2.2, -1.1, 0.4, 1.2, 2, 3.1, 5)]]

    smooth_lower_sums_op(divs, foos, FHD, .1)
