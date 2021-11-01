from src.anamator.basic_func import *
from src.anamator.objects import *
from math import pi, sin, cos, exp


def plot_mormul(filename):
    grad_norm = Function(lambda t: (1/2 + sin(t)*sin(2*t)/2 + cos(t)*cos(2*t)/2)**.5)
    grad_max = Function(lambda t: (1/2 + sin(t)*sin(2*t)/2 + cos(t)*cos(2*t)/2)**.5).sup((0, 2*pi))
    color = lambda t: (0xFF, 0xFF - 255*grad_norm(t)/grad_max,  0xFF - 255*grad_norm(t)/grad_max, 1)

    def pipe(t):
        k = 100

        # [PipeInstance(ParametricBlittingSettings(thickness=2, blur=0),
        #               ParametricObject(lambda x: cos(x) / 2 + cos(2 * x) / 4,
        #                                lambda x: sin(x) / 2 - sin(2 * x) / 4),
        #               interval_of_param=[0, 2 * pi]),
        t2 = 2*pi - 2*t
        return [PipeInstance(ParametricBlittingSettings(thickness=6, blur=0),
                      ParametricObject(lambda x: cos(x) / 2 + cos(2 * x) / 4,
                                       lambda x: sin(x) / 2 - sin(2 * x) / 4),
                      interval_of_param=[0, 2 * pi]),
                PipeInstance(ParametricBlittingSettings(color='pink', thickness=6, blur=0),
                                   PolygonalChain([(cos(t)/2 + cos(2*t)/4, sin(t)/2 - sin(2*t)/4),
                                                   (cos(t+pi)/2 + cos(2*t)/4, sin(t+pi)/2 - sin(2*t)/4)])),

                      PipeInstance(ParametricBlittingSettings(color='gray', thickness=6, blur=0),
                                   ParametricObject(lambda x: cos(x)/4 + cos(t)*.5,
                                                    lambda x: sin(x)/4 + sin(t)*.5),
                                   interval_of_param=[0, 2*pi]),
                      # PipeInstance(ParametricBlittingSettings(color='gray', thickness=2, blur=0),
                      #              ParametricObject(lambda x: cos(x) / 4 + cos(t+pi) * .5,
                      #                               lambda x: sin(x) / 4 + sin(t+pi) * .5),
                      #              interval_of_param=[0, 2 * pi]),
                      PipeInstance(ParametricBlittingSettings(color='white', thickness=6, blur=0),
                                   ParametricObject(lambda x: cos(x)*.75,
                                                    lambda x: sin(x)*.75),
                                   interval_of_param=[0, 2 * pi]),
                      PipeInstance(BitmapBlittingSettings(blur=0), BitmapDisk(15, 'white', 1),
                                   blitting_type='bitmap', center=(cos(t)/2 + cos(2*t)/4, sin(t)/2 - sin(2*t)/4)),
                      # PipeInstance(BitmapBlittingSettings(blur=0), BitmapDisk(6, 'white', 1),
                      #              blitting_type='bitmap', center=(cos(t+pi)/2 + cos(2*t)/4, sin(t+pi)/2 - sin(2*t)/4)),

                      # PipeInstance(ParametricBlittingSettings(color='pink', thickness=1, blur=0),
                      #              PolygonalChain([(cos(t)*.5, sin(t)*.5), (-cos(t)*.5, -sin(t)*.5)])),

                # PipeInstance(ParametricBlittingSettings(color='pink'),
                #              PolygonalChain([(cos(t2) / 2 + cos(2 * t2) / 4, sin(t2) / 2 - sin(2 * t2) / 4),
                #                              (cos(t2 + pi) / 2 + cos(2 * t2) / 4, sin(t2 + pi) / 2 - sin(2 * t2) / 4)])),

                PipeInstance(ParametricBlittingSettings(color='gray', thickness=6, blur=0),
                             ParametricObject(lambda x: cos(x) / 4 + cos(t2) * .5,
                                              lambda x: sin(x) / 4 + sin(t2) * .5),
                             interval_of_param=[0, 2 * pi]),
                # PipeInstance(ParametricBlittingSettings(color='gray', thickness=2, blur=0),
                #              ParametricObject(lambda x: cos(x) / 4 + cos(t2 + pi) * .5,
                #                               lambda x: sin(x) / 4 + sin(t2 + pi) * .5),
                #              interval_of_param=[0, 2 * pi]),
                PipeInstance(ParametricBlittingSettings(color='white', thickness=6, blur=0),
                             ParametricObject(lambda x: cos(x) * .75,
                                              lambda x: sin(x) * .75),
                             interval_of_param=[0, 2 * pi]),
                PipeInstance(BitmapBlittingSettings(blur=0), BitmapDisk(15, 'white', 1),
                             blitting_type='bitmap', center=(cos(t2) / 2 + cos(2 * t2) / 4, sin(t2) / 2 - sin(2 * t2) / 4)),
                # PipeInstance(BitmapBlittingSettings(blur=0), BitmapDisk(12, 'white', 1),
                #              blitting_type='bitmap',
                #              center=(cos(t2 + pi) / 2 + cos(2 * t2) / 4, sin(t2 + pi) / 2 - sin(2 * t2) / 4)),
                #
                # PipeInstance(ParametricBlittingSettings(color='pink', thickness=1, blur=0),
                #              PolygonalChain([(cos(t2) * .5, sin(t2) * .5), (-cos(t2) * .5, -sin(t2) * .5)]))
                      ]

    bounds = lambda t: np.array([(-1.72, 1.72), (-.88, .88)])
    differentials = [make_periodic(lambda t: Gaussian(.5, 70, None)(t))]
    differentials = [Gaussian(.5, 70, None)]
    differentials = [lambda t: 1.1*pi*make_periodic(PredefinedSettings.slow_differential)(t)]
    differentials = [lambda t: 1.1*pi]

    animation = AnimationPipeline(pipe, bounds, differentials)
    animation.render(filename, PipelineSettings(duration=1, x_padding=200, y_padding=200, resolution=F4K),
                     speed=.2, id_='mormul', png_filename='mormul_colors.png')


if __name__ == '__main__':
    plot_mormul('deltoid.mp4')