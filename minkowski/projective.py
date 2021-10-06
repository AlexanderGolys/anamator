import numpy as np

from src.anamator.basic_func import *
from src.anamator.objects import *


F4K = (3840, 2160)
RADIUS_FOO = PredefinedSettings.radius_func_creator(24, 16)
CIRCLES_THIC = 8
PADDING = 200


def doubling_vectors(filename):
    def frame_generator(*t):
        vectors = [[(0, 0), (1, 2)],
                   [(0, 0), (-.5, -.8)],
                   [(0, 0), (-1, 1.6)],
                   [(0, 0), (.3, -1.3)]]
        colors = ['indian red', 'champagne pink', 'columbia blue', 'baby powder']

        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-5, 5), (-5, 5))
        for t, v, color in zip(t, vectors, colors):
            frame.axis_surface.blit_vector(v[0], (1+t)*np.array(v[1]),
                                           ParametricBlittingSettings(thickness=5, color=color, blur=2),
                                           BitmapBlittingSettings(blur=0), position_correction=[0, 0],
                                           angle_correction=0, arrow_path='src//anamator//img//arrow.png')
        frame.blit_axes(objects.ParametricBlittingSettings(blur=0, thickness=3))
        frame.blit_axis_surface()
        return frame

    differentials = [PredefinedSettings.exp_differential] + [lambda x: 0]*3
    animator = MultiDifferentialAnimation(frame_generator, *differentials)
    animator.render(filename, RenderSettings(), speed=.25)


def vector_to_lines(filename):
    def frame_generator(*t):
        vectors = [[(0, 0), (1, 2)],
                   [(0, 0), (-.5, -.8)],
                   [(0, 0), (-1, 1.6)],
                   [(0, 0), (.3, -1.3)]]
        colors = ['green 1', 'indian red', 'blue', 'gray']

        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-4.3, 4.3), (-2.2, 2.2))
        for t, v, color in zip(t, vectors, colors):
            if t < .5:
                frame.axis_surface.blit_vector(v[0], 2*(.5-t)*np.array(v[1]),
                                               ParametricBlittingSettings(thickness=5, color=color, blur=2),
                                               BitmapBlittingSettings(blur=0), position_correction=[0, 0],
                                               angle_correction=0, arrow_path='src//anamator//img//arrow.png')
            else:
                line = Function(lambda x: v[1][1]/v[1][0]*x)
                frame.axis_surface.blit_parametric_object(line, interval_of_param=2*(t-.5)*np.array([-5.1, 5.1]),
                                                          settings=ParametricBlittingSettings(thickness=3, color=color))
        frame.blit_axes(objects.ParametricBlittingSettings(blur=0, thickness=3))
        frame.blit_axis_surface()
        return frame

    differentials = [Gaussian(k, 70, None) for k in np.linspace(.3, .7, 4)]
    animator = MultiDifferentialAnimation(frame_generator, *differentials)
    animator.render(filename, RenderSettings(), speed=.25)


def lines_to_circle(filename):
    def frame_generator(circle_t, *t):
        vectors = [[(0, 0), (1, 2)],
                   [(0, 0), (-.5, -.8)],
                   [(0, 0), (-1, 1.6)],
                   [(0, 0), (.3, -1.3)]]
        colors = ['green 1', 'indian red', 'blue', 'gray']

        def radius(x):
            if x < .5:
                return 0
            if x < .75:
                return 48*(x-.5)
            return -16*(x-.75) + 12

        frame = OneAxisFrame(FHD)
        frame.add_axis_surface((-4.3, 4.3), (-2.2, 2.2))
        circle = ParametricObject(lambda x: cos(2*pi*x), lambda x: sin(2*pi*x))
        frame.axis_surface.blit_parametric_object(circle, interval_of_param=(0, circle_t),
                                                  settings=ParametricBlittingSettings(thickness=3, blur=0))
        for t, v, color in zip(t, vectors, colors):
            normal_v = np.array(v[1])/np.linalg.norm(v[1])
            if normal_v[1] < 0:
                normal_v *= -1
            line = Function(lambda x: v[1][1]/v[1][0]*x)
            frame.axis_surface.blit_parametric_object(line, interval_of_param=SingleAnimation.blend_lists([[-5.1, 5.1], [normal_v[0], normal_v[0]]], t),
                                                      settings=ParametricBlittingSettings(thickness=3, color=color))
            dot = objects.BitmapDisk(round(radius(t)), color, 1)
            frame.axis_surface.blit_bitmap_object(normal_v, dot,
                                                  BitmapBlittingSettings(), surface_coordinates=True)
        frame.blit_axes(objects.ParametricBlittingSettings(blur=0, thickness=3))
        frame.blit_axis_surface()
        return frame

    differentials = [Gaussian(k, 70, None) for k in np.linspace(.2, .7, 5)]
    animator = MultiDifferentialAnimation(frame_generator, *differentials)
    animator.render(filename, RenderSettings(), speed=.25)


def line_race(filename, steps=100):
    video = Film(24, FHD, 'race')
    for i, h in enumerate(np.linspace(.5, 5, steps)):
        print(f'{i}/{steps}')
        center = np.array((h**2, 2*h**2))
        line1 = Function(lambda x: 2*x)
        line2 = Function(lambda x: 2*x + 1)
        frame = OneAxisFrame(FHD)
        y_bounds = np.array([center[1]-2.7, center[1]+2.7])
        x_bounds = np.array([center[0]-4.8, center[0]+4.8])
        frame.add_axis_surface(x_bounds, y_bounds)
        for n in range(int(x_bounds[0]), int(x_bounds[1]) + 1):
            grid_line = ParametricObject(create_const(n), lambda y: y)
            frame.axis_surface.blit_parametric_object(grid_line,
                                                      ParametricBlittingSettings(thickness=2, color='gray', blur=0),
                                                      interval_of_param=y_bounds,
                                                      queue=True)
        for n in range(int(y_bounds[0]), int(y_bounds[1]) + 1):
            grid_line = ParametricObject(lambda x: x, create_const(n))
            frame.axis_surface.blit_parametric_object(grid_line,
                                                      ParametricBlittingSettings(sampling_rate=10, thickness=2, color='gray', blur=0),
                                                      interval_of_param=x_bounds,
                                                      queue=True)
        frame.axis_surface.blit_parametric_queue()
        frame.axis_surface.blit_axes(settings=ParametricBlittingSettings(thickness=3, blur=0))
        frame.axis_surface.blit_parametric_object(line2, ParametricBlittingSettings(color='gray'))
        frame.axis_surface.blit_parametric_object(line1, interval_of_param=(x_bounds[0], center[0] + 2*i**2/steps**2))
        frame.blit_axis_surface()
        video.add_frame(frame, save_ram=True)
    video.render(filename, True)


def line_crossings(filename):
    color = 'champagne pink'
    color2 = 'gray'
    radius_peak = 12
    radius_final = 8

    x_start, x_spread = -1.5, 3
    y_start, y_spread = -1.5, 3
    bounds = lambda *t: [[x_start, x_start+x_spread], [y_start, y_start+y_spread]]
    functions = [lambda x: -x, lambda x: -2*x, lambda x: -.5*x, lambda x: -3*x, lambda x: .25*x,
                 lambda x: -10*x, lambda x: .5*x, lambda x: .75*x, lambda x: .88*x, lambda x: 20*x]

    def pipe_instance_creator(foo, i, t):
        return PipeInstance(ParametricBlittingSettings(thickness=7, color=color, sampling_rate=20), Function(foo),
                            interval_of_param=t[i]*np.array((x_start, x_start+x_spread)))

    radius = PredefinedSettings.radius_func_creator(radius_peak, radius_final)

    def crossing_point(foo):
        f = Function(lambda x: 2*x + 1 - foo(x))
        return f.zeros(bounds(0)[0], precision=1000)[0], foo(f.zeros(bounds(0)[0], precision=10000)[0])

    pipe = lambda *t: [PipeInstance(ParametricBlittingSettings(blur=0, thickness=3), blitting_type='axis'),
                       PipeInstance(ParametricBlittingSettings(thickness=7, color=color2), Function(lambda x: 2*x+1))]\
                       + [pipe_instance_creator(foo, i, t) for i, foo in enumerate(functions)] \
                       + [PipeInstance(BitmapBlittingSettings(), BitmapDisk(round(radius(t[i])), color, 1),
                                       blitting_type='bitmap', center=crossing_point(foo)) for i, foo in enumerate(functions)]
    differentials = [Gaussian(m, 70, None) for m in np.linspace(0.25, 0.75, len(functions))]
    animation = AnimationPipeline(pipe, bounds, differentials)
    animation.render(filename, PipelineSettings())


def rotate_projective_line(filename):
    color = 'champagne pink'
    color2 = 'gray'
    radius = 8

    angle = math.atan(2)

    x_start, x_spread = -1.5, 3
    y_start, y_spread = -1.5, 3
    bounds = [[x_start, x_start + x_spread], [y_start, y_start + y_spread]]
    f_bounds = lambda t: [[x_start, x_start + x_spread], [y_start, y_start + y_spread]]
    functions = [lambda x: -x, lambda x: -2 * x, lambda x: -.5 * x, lambda x: -3 * x, lambda x: .25 * x,
                 lambda x: -10 * x, lambda x: .5 * x, lambda x: .75 * x, lambda x: .88 * x, lambda x: 20 * x]

    points = [AnalyticGeometry.crossing_point_of_functions(lambda x: 2*x + 1, foo, bounds[0]) for foo in functions]

    def pipe(t):
        shift = np.array([0, -t])
        rot_points = [AnalyticGeometry.rotate_point(p, -angle*t) + shift for p in points]
        return [PipeInstance(ParametricBlittingSettings(blur=0, thickness=3), blitting_type='axis')] \
            + [PipeInstance(ParametricBlittingSettings(thickness=7, color=color, sampling_rate=20, blur=2),
                            Function(foo)) for foo in functions] \
            + [PipeInstance(ParametricBlittingSettings(color=(0, 0, 0, t),
                                                       sampling_rate=3, thickness=0, blur=0),
                            FilledObject(Function(const=2), Function(const=-2), interval=[-2, 2]))] \
            + [PipeInstance(ParametricBlittingSettings(thickness=7, color=color2, blur=0),
                            AnalyticGeometry.line_between_points(rot_points[0], rot_points[-1], [-10, 10]))] \
            + [PipeInstance(BitmapBlittingSettings(), [BitmapDisk(radius, color, 1)]*len(functions),
                            blitting_type='bitmap', centers=rot_points)]
    differentials = [Gaussian(.5, 50, None)]
    animation = AnimationPipeline(pipe, f_bounds, differentials)
    animation.render(filename, PipelineSettings())


def circles_collapsing(filename, speed):
    r1, r2 = 100, 200
    # c1, c2 = np.array((200, 150)), np.array((1000, 500))
    c1, c2 = np.array((-50, 0)), np.array((40, 0))

    between = (c1 + c2)//2

    radius = lambda r, t: int(max(r*(1 - 2*t), 0))
    number_scale = lambda t: PredefinedSettings.radius_func_creator(200, 100)(max((2*t - 1), 0))/300
    dot_scale = lambda t: .5*int(t < .5)

    def pipe(t):
        return [PipeInstance(BitmapBlittingSettings(), ImageObject('minkowski//img//product.png', scale=dot_scale(t)),
                             blitting_type='bitmap',
                             center=between),
                PipeInstance(BitmapBlittingSettings(), BitmapCircle(radius(r1, t), 'white', 7, 1),
                             blitting_type='bitmap',
                             center=SingleAnimation.blend_lists([c1, between], min(2*t, 1))),
                PipeInstance(BitmapBlittingSettings(), BitmapCircle(radius(r2, t), 'white', 7, 1),
                             blitting_type='bitmap',
                             center=SingleAnimation.blend_lists([c2, between], min(2 * t, 1))),
                PipeInstance(BitmapBlittingSettings(), ImageObject('minkowski//img//number.png', scale=number_scale(t)),
                             blitting_type='bitmap',
                             center=between)]
    differentials = [Gaussian(.5, 50, None)]
    animation = AnimationPipeline(pipe, lambda t: [(-100, 100), (-100, 100)], differentials)
    animation.render(filename, PipelineSettings(), speed=speed)


def different_configurations(number, filename):
    def color_growing(t):
        return 'gray' if t > .01 else 'black'

    def color_collapsing(t):
        return 'gray' if t < .99 else 'black'

    first_r1 = 50
    first_r2 = 70
    second_r1 = 40
    second_r2 = 55

    first_center1 = -100
    first_center2 = -10
    second_center1 = -115
    second_center2 = -30

    third_r = 30

    # distinct to single
    if number == 1:
        r1 = first_r1
        r2 = first_r2
        c1_old, c2_old = np.array((first_center1, 0)), np.array((first_center2, 0))
        between = (c1_old + c2_old) // 2
        radius_vector = np.array([1, 1]) * (r1 + r2) / (2 * 2 ** .5)
        pipe = lambda t1, t2: [PipeInstance(ParametricBlittingSettings(color=color_growing(t2)),
                                            PolygonalChain([between, between + t2*radius_vector])),
                               PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                            Ellipse(SingleAnimation.blend_lists([c1_old, between], t1),
                                                    (1-t1)*r1 + t1*(r1 + r2)/2,
                                                    (1-t1)*r1 + t1*(r1 + r2)/2)),
                               PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                            Ellipse(SingleAnimation.blend_lists([c2_old, between], t1),
                                                    (1 - t1) * r2 + t1 * (r1 + r2) / 2,
                                                    (1 - t1) * r2 + t1 * (r1 + r2) / 2)),
                               ]
        bounds = lambda t1, t2: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.4, 50, None), Gaussian(.7, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K), speed=.25)

    # single to distinct
    if number == 2:
        r1 = second_r1
        r2 = second_r2
        c1, c2 = np.array((second_center1, 0)), np.array((second_center2, 0))
        between = (np.array((first_center1, 0)) + np.array((first_center2, 0)))/2
        radius_vector = np.array([1, 1]) * (r1 + r2) / (2 * 2 ** .5)
        pipe = lambda t1, t2: [PipeInstance(ParametricBlittingSettings(color=color_collapsing(t1)),
                                            PolygonalChain([between, between + (1 - t1) * radius_vector])),
                               PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                            Ellipse(SingleAnimation.blend_lists([between, c1], t2),
                                                    t2*r1 + (1 - t2)*(r1 + r2)/2,
                                                    t2*r1 + (1 - t2)*(r1 + r2)/2)),
                               PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                            Ellipse(SingleAnimation.blend_lists([between, c2], t2),
                                                    t2*r2 + (1 - t2)*(r1 + r2)/2,
                                                    t2*r2 + (1 - t2)*(r1 + r2)/2)),
                               ]
        bounds = lambda t1, t2: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.3, 50, None), Gaussian(.6, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K), speed=.25)

    # distinct to orthogonal
    if number == 3:
        r1 = second_r1
        r2 = second_r2
        c1_old, c2_old = np.array((second_center1, 0)), np.array((second_center2, 0))
        c1_end = (c1_old + c2_old) / 2 - np.array([(r1**2 + r2**2)**.5/2, 0])
        c2_end = (c1_old + c2_old) / 2 + np.array([(r1 ** 2 + r2 ** 2) ** .5 / 2, 0])

        pipe = lambda t1: [PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                        Ellipse(SingleAnimation.blend_lists([c1_old, c1_end], t1), r1, r1)),
                           PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                        Ellipse(SingleAnimation.blend_lists([c2_old, c2_end], t1), r2, r2))]
        bounds = lambda t1: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.5, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K), speed=.25)

        # orthogonal to tangent
    if number == 4:
        r1 = second_r1
        r2 = second_r2
        c1_old, c2_old = np.array((second_center1, 0)), np.array((second_center2, 0))
        c1_start = (c1_old + c2_old) / 2 - np.array([(r1 ** 2 + r2 ** 2) ** .5 / 2, 0])
        c2_start = (c1_old + c2_old) / 2 + np.array([(r1 ** 2 + r2 ** 2) ** .5 / 2, 0])

        c1_end = (c1_old + c2_old) / 2 - np.array([(r1 + r2) / 2, 0])
        c2_end = (c1_old + c2_old) / 2 + np.array([(r1 + r2) / 2, 0])

        pipe = lambda t1: [PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                        Ellipse(SingleAnimation.blend_lists([c1_start, c1_end], t1), r1, r1)),
                           PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                        Ellipse(SingleAnimation.blend_lists([c2_start, c2_end], t1), r2, r2))]
        bounds = lambda t1: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.5, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K),
                         speed=.25)

    # tangent to the point
    if number == 5:
        r1 = second_r1
        r2 = second_r2
        c1_old, c2_old = np.array((second_center1, 0)), np.array((second_center2, 0))
        c1_old = (c1_old + c2_old) / 2 - np.array([(r1 + r2) / 2, 0])
        c2_old = (c1_old + c2_old) / 2 + np.array([(r1 + r2) / 2, 0])
        c1 = (c1_old + c2_old) / 2 - np.array([(r1 + r2) / 2, 0])
        c2 = (c1_old + c2_old) / 2 + np.array([(r1 + r2) / 2, 0])

        pipe = lambda t1, t2: [PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                            Ellipse(c1, r1, r1)),
                               PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                            Ellipse(c2, (1-t1)*r2, (1-t1)*r2)),
                               PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(t2), 'white', 1),
                                            blitting_type='bitmap', center=c2)]
        bounds = lambda t1, t2: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.3, 50, None), Gaussian(.7, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K),
                         speed=.25)

    # point lying on circle
    if number == 6:
        r1 = second_r1
        r2 = second_r2
        c1_old, c2_old = np.array((-50, 0)), np.array((100, 0))

        c1 = (c1_old + c2_old) / 2 - np.array([(r1 + r2) / 2, 0])
        c2 = (c1_old + c2_old) / 2 + np.array([(r1 + r2) / 2, 0])
        c2_end = c1 + np.array([r1, 0])

        pipe = lambda t: [PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                       Ellipse(c1, r1, r1)),
                          PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(t), 'white', 1),
                                       blitting_type='bitmap', center=SingleAnimation.blend_lists([c2, c2_end], t))]
        bounds = lambda t: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.3, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K),
                         speed=.25)

    # two points
    if number == 7:
        r1 = second_r1
        r2 = second_r2
        c1_old, c2_old = np.array((-50, 0)), np.array((100, 0))

        c1 = (c1_old + c2_old) / 2 - np.array([(r1 + r2) / 2, 0])
        c2 = (c1_old + c2_old) / 2 + np.array([(r1 + r2) / 2, 0])
        c2_end = c1 + np.array([r1, 0])

        between = (c1 + c2_end)/2

        pipe = lambda t1, t2, t3: [PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0),
                                                Ellipse(c1, (1-t1)*r1, (1-t1)*r1)),
                                   PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(t2), 'white', 1),
                                                blitting_type='bitmap',
                                                center=c1),
                                   PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(t2), 'white', 1),
                                                blitting_type='bitmap', center=c2_end),
                                   PipeInstance(ParametricBlittingSettings(color=color_growing(t3)),
                                                PolygonalChain([SingleAnimation.blend_lists([between, c1], t3),
                                                               SingleAnimation.blend_lists([between, c2_end], t3)]))
                                   ]
        bounds = lambda t1, t2, t3: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.3, 50, None), Gaussian(.5, 50, None), Gaussian(.7, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K),
                         speed=.25)

    # creating circle and line
    y1_shift = np.array([0, -50])
    y2_shift = np.array([0, 50])

    if number == 8:
        r1 = second_r1
        r2 = second_r2
        c1_old, c2_old = np.array((-50, 0)), np.array((100, 0))

        c1 = (c1_old + c2_old) / 2 - np.array([(r1 + r2) / 2, 0])
        c2 = (c1_old + c2_old) / 2 + np.array([(r1 + r2) / 2, 0])
        c2 = c1 + np.array([r1, 0])

        point_c1 = c1 + y1_shift
        point_c2 = c2 + y1_shift

        circle_c = c1 + y2_shift
        line_c = c2 + y2_shift

        r = third_r

        pipe = lambda t1, t2: [PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(1), 'white', 1),
                                            blitting_type='bitmap',
                                            center=SingleAnimation.blend_lists([c2, point_c2], t1)),
                               PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(1), 'white', 1),
                                            blitting_type='bitmap',
                                            center=SingleAnimation.blend_lists([c1, point_c1], t1)),
                               PipeInstance(ParametricBlittingSettings(color='gray'),
                                            PolygonalChain([SingleAnimation.blend_lists([c1, point_c1], t1),
                                                            SingleAnimation.blend_lists([c2, point_c1], t2)])),

                               PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(1-t2), 'white', 1),
                                            blitting_type='bitmap',
                                            center=SingleAnimation.blend_lists([c2, line_c], t1)),
                               PipeInstance(BitmapBlittingSettings(), BitmapDisk(RADIUS_FOO(1-t2), 'white', 1),
                                            blitting_type='bitmap',
                                            center=SingleAnimation.blend_lists([c1, circle_c], t1)),
                               PipeInstance(ParametricBlittingSettings(thickness=CIRCLES_THIC, blur=0,
                                                                       color=color_growing(t2)),
                                            Ellipse(circle_c, t2*r, t2*r)),
                               PipeInstance(ParametricBlittingSettings(color='white'),
                                            PolygonalChain([SingleAnimation.blend_lists([c1, circle_c], t1),
                                                            SingleAnimation.blend_lists([c2, line_c], t2)])),
                               PipeInstance(ParametricBlittingSettings(color='white'),
                                            PolygonalChain([SingleAnimation.blend_lists([line_c, line_c+y1_shift], t2),
                                                            SingleAnimation.blend_lists([line_c, line_c+y2_shift], t2)])),
                               ]
        bounds = lambda t1, t2: ((-172, 172), (-88, 88))
        differentials = [Gaussian(.3, 50, None), Gaussian(.7, 50, None)]
        animation = AnimationPipeline(pipe, bounds, differentials)
        animation.render(filename, PipelineSettings(x_padding=PADDING, y_padding=PADDING, resolution=F4K),
                         speed=.25)


if __name__ == '__main__':
    # objects.ImageObject('src//anamator//img//arrow.png')
    # doubling_vectors('doubling.mp4')
    # vector_to_lines('vec_to_line.mp4')
    # lines_to_circle('lines_to_circle.mp4')
    # line_race('line_race.mp4', 240)
    # line_crossings('crossing.mp4')
    # rotate_projective_line('minkowski//renders//rotate_projective_line.mp4')
    # circles_collapsing('minkowski//renders//collapsing.mp4', .25)

    # different_configurations(1, 'minkowski//final//conf1.mp4')
    # different_configurations(2, 'minkowski//final//conf2.mp4')
    # different_configurations(3, 'minkowski//final//conf3.mp4')
    different_configurations(4, 'minkowski//final//conf4.mp4')
    different_configurations(5, 'minkowski//final//conf5.mp4')
    different_configurations(6, 'minkowski//final//conf6.mp4')
    different_configurations(7, 'minkowski//final//conf7.mp4')
    different_configurations(8, 'minkowski//final//conf8.mp4')

