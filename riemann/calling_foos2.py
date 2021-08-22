from riemann.riemann_main import *


if __name__ == '__main__':
    # intervals_into_divisions([.1, .3, .45, .7, .9], .3, FHD, filename='intervals_into_points.mp4', slow=True)
    # intervals_into_divisions([.1, .3, .45, .7, .9], .3, FHD, filename='intervals_into_points_fast.mp4', slow=False)
    # gradient4d()
    # division_ = [-4, -3.5, -3, -2, -1.7, -1.3, -1, -.3, .5, 1.5, 2, 2.2, 2.4, 2.6, 2.9, 3, 3.8, 4]
    # smooth_animate_change_to_recs(division_, lambda x: math.exp(x / 10) * math.sin(x) ** 2 + 0.2, FHD, .15,
    #                               'change_to_rec.mp4')
    flying_recs(speed=.25)