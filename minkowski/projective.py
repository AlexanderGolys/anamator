import numpy as np

from src.anamator.basic_func import *
from src.anamator.objects import *


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

    x_start, x_spread = -1.5, 3
    y_start, y_spread = -1.5, 3
    bounds = [[x_start, x_start + x_spread], [y_start, y_start + y_spread]]
    f_bounds = lambda t: [[x_start, x_start + x_spread], [y_start, y_start + y_spread]]
    functions = [lambda x: -x, lambda x: -2 * x, lambda x: -.5 * x, lambda x: -3 * x, lambda x: .25 * x,
                 lambda x: -10 * x, lambda x: .5 * x, lambda x: .75 * x, lambda x: .88 * x, lambda x: 20 * x]

    points = [AnalyticGeometry.crossing_point_of_functions(lambda x: 2*x + 1, foo, bounds[0]) for foo in functions]

    def pipe(t):
        rot_points = [AnalyticGeometry.rotate_point(p, 2*pi*t) for p in points]
        return [PipeInstance(ParametricBlittingSettings(blur=0, thickness=3), blitting_type='axis')] \
            + [PipeInstance(ParametricBlittingSettings(thickness=7, color=color, sampling_rate=20),
                            Function(foo), queue=True) for foo in functions] \
            + [PipeInstance(BitmapBlittingSettings(), BitmapDisk(10, 'black', 1-t), blitting_type='bitmap',
                            center=(0, 0))] \
            + [PipeInstance(ParametricBlittingSettings(thickness=7, color=color2),
                            AnalyticGeometry.line_between_points(rot_points[0], rot_points[-1], bounds[0]))] \
            + [PipeInstance(BitmapBlittingSettings(), [BitmapDisk(radius, color, 1)]*len(functions),
                            blitting_type='bitmap', center=rot_points)]
    differentials = [Gaussian(.5, 70, None)]
    animation = AnimationPipeline(pipe, f_bounds, differentials)
    animation.render(filename, PipelineSettings())


if __name__ == '__main__':
    # objects.ImageObject('src//anamator//img//arrow.png')
    # doubling_vectors('doubling.mp4')
    # vector_to_lines('vec_to_line.mp4')
    # lines_to_circle('lines_to_circle.mp4')
    # line_race('line_race.mp4', 240)
    # line_crossings('crossing.mp4')
    rotate_projective_line('rotate_projective_line.mp4')