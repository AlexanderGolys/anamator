import numpy as np
import math
import os

from animator import basic_func, objects


def epoch_frame(target_func, approximators_list, generate_png=False, top_n=5):
    """
    Plotting family of functions belonging to single epoch and the target function.

    Args:
        target_func (function): Function to be approximated.
        approximators_list (list of functions): Approximators from one epoch.
        generate_png (bool, optional): Saves the image to 'frame.png' if True.
        top_n (int): How many best functions should be printed.

    Returns:
        basic_func.OneAxisFrame: Generated frame.
    """
    frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)
    target_function = objects.Function(target_func)
    approximators = [objects.Function(foo) for foo in approximators_list[:top_n]]

    settings_target_function = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'white'
    }
    settings_approximators = {
        'sampling rate': 3,
        'thickness': 4,
        'blur': 2,
        'color': 'light gray'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 2,
        'blur': 1,
        'color': 'white'
    }

    frame.add_axis_surface(x_bounds=(0, 1), y_bounds=(-2, 2))
    frame.blit_axes(settings_axes, x_only=True)
    for foo in approximators:
        frame.axis_surface.blit_parametric_object(foo, settings_approximators, queue=True)
    frame.axis_surface.blit_parametric_queue()
    frame.blit_parametric_object(target_function, settings_target_function)
    frame.blit_axis_surface()
    if generate_png:
        frame.generate_png(f'frame.png')
    return frame


def make_film(target_func, epochs, filename='genetic.mp4', fps=1, resolution=(1280, 720), step=1, top_n=5,
              number_of_frames=5, save_ram=True, id='', read_only=False):
    """
    Generates the video illustrating function approximation by genetic algorithm.

    Args:
        target_func (function): Function to be approximated.
        epochs (list): List of epochs, where epoch is a list of approximators belonging to that epoch.
        filename (str): Target filename.
        fps (int): Frames per second.
        resolution: Resolution of the video. Should be the same as frame resolution.
        step (int): Skip every step epochs in the video.
        top_n (int): How many best function should be plotted.
        number_of_frames (int): IF not None, step is to generate specific number of frames
        save_ram (bool): Render in save_ram mode.
        id (str): Film id.
        read_only (bool): Render already saved frames.
    """

    video = basic_func.Film(fps, resolution, id=id)

    if number_of_frames is not None:
        number_of_frames -= 1

    if read_only:
        video.frame_counter = number_of_frames
        video.render(filename, save_ram=True)
        return

    if number_of_frames is not None:
        step = len(epochs)//number_of_frames

    print(f'step: {step}, frames: {len(epochs[::step])}')
    for i, epoch in enumerate(epochs[::step]):
        video.add_frame(epoch_frame(target_func, epoch, top_n=top_n), save_ram=save_ram)
        print(i)
    video.render(filename, save_ram=save_ram)


def make_polynomial(coef):
    return lambda x: sum(a*x**i for i, a in enumerate(coef))


def make_random_generation(n=10):
    return [make_polynomial([np.random.normal() for _ in range(10)]) for _ in range(n)]


def make_random_epochs(n=10):
    return [make_random_generation() for _ in range(n)]


class Approximant:
    def __init__(self, length, values=None, std=None, default_std=1):

        self.length = length
        self.values = np.random.normal(0, default_std, (length,))
        self.std = np.abs(np.random.normal(0, default_std, (length,)))
        if values is not None:
            self.values = values
            self.std = std
        self.temp_fitness_rank = None
        self.temp_diversity_rank = None

    def make_polynomial(self):
        return lambda x: sum(a*x**i for i, a in enumerate(self.values))

    def sex(self, other):
        t = np.random.random()
        return Approximant(self.length, values=t*self.values + (1-t)*other.values,
                           std=t*self.std + (1-t)*other.std, )

    def mutate(self):
        self.values += np.random.normal(0, self.std, (self.length,))
        self.std += np.random.normal(np.zeros((self.length,)), self.std, (self.length,))
        self.std = np.abs(self.std)
        return self

    def fitness(self, target_foo, norm='sup', sampling_rate=100):
        if norm == 'sup':
            return 1/(max([abs(target_foo(x) - self.make_polynomial()(x)) for x in np.linspace(0, 1, sampling_rate)])+1)
        if norm == 'int':
            return 1/(sum([abs(target_foo(x) - self.make_polynomial()(x)) for x in np.linspace(0, 1, sampling_rate)])
                      / sampling_rate + 1)

    def diversity(self, new_population):
        return np.mean(np.array([np.linalg.norm(self.values - x.values) for x in new_population]))

    def __str__(self):
        return ' '.join(map(str, self.values))


class Population:
    def __init__(self, size, unit_length, target_function, society=None, default_std=1, p_c=.1, metric='int'):
        self.size = size
        self.p_c = p_c
        self.unit_length = unit_length
        self.society = society if society is not None else [Approximant(unit_length, default_std=default_std) for _ in range(size)]
        self.target_function = target_function
        self.metric = metric

    def group_sex(self, selection_type='proportional', save_king=True):
        self.society.sort(key=lambda x: x.fitness(self.target_function, norm=self.metric), reverse=True)

        for i, x in enumerate(self.society):
            x.temp_fitness_rank = i

        if selection_type == 'diversity rank':
            distribution = [(1 - self.p_c) ** k * self.p_c for k in range(self.size - 1)] + \
                           [(1 - self.p_c) ** (self.size - 1)]
            parents = np.random.choice(self.society, 2, replace=True, p=distribution)
            new_population = [parents[0].sex(parents[1]).mutate()]

            while len(new_population) != self.size:
                self.society.sort(key=lambda guy: guy.diversity(new_population), reverse=True)
                for i, x in enumerate(self.society):
                    x.temp_diversity_rank = i
                self.society.sort(key=lambda guy: guy.temp_fitness_rank**2 + guy.temp_diversity_rank**2)
                parents = np.random.choice(self.society, 2, replace=True, p=distribution)
                new_population += [parents[0].sex(parents[1]).mutate()]
                if save_king:
                    new_population[-1] = max(self.society, key=lambda x: x.fitness(self.target_function,
                                                                                   norm=self.metric))
            return Population(self.size, self.unit_length, self.target_function, new_population, metric=self.metric)

        if selection_type == 'proportional':
            sum_of_fitness = sum([x.fitness(self.target_function, norm=self.metric) for x in self.society])
            distribution = np.array([x.fitness(self.target_function, norm=self.metric)/sum_of_fitness for x in self.society])

        else:
            distribution = np.array([(1-self.p_c)**k*self.p_c for k in range(self.size-1)]
                                    + [(1-self.p_c)**(self.size-1)])

        # print(np.array(self.society).size, distribution.size)
        new_mothers = np.random.choice(np.array(self.society), self.size, replace=True, p=distribution)
        new_fathers = np.random.choice(self.society, self.size, replace=True, p=distribution)
        new_population = [x.sex(y).mutate() for x, y in zip(new_fathers, new_mothers)]
        if save_king:
            new_population[0] = max(self.society, key=lambda x: x.fitness(self.target_function, norm=self.metric))
        return Population(self.size, self.unit_length, self.target_function, new_population, metric=self.metric)

    def census(self):
        self.society.sort(key=lambda x: x.fitness(self.target_function, norm=self.metric), reverse=True)
        return [make_polynomial(x.values) for x in self.society]

    def get_best_fitness(self):
        self.society.sort(key=lambda guy: guy.fitness(self.target_function, norm=self.metric), reverse=True)
        return self.society[0].fitness(self.target_function, norm=self.metric)

    def __str__(self):
        return '\n'.join(map(str, self.society))


def genetic_algorithm(target_function, population_size, unit_length, epochs, selection_type='rank', default_std=1,
                      save_king=True, p_c=.1, metric='int'):
    populations = [Population(population_size, unit_length, target_function, default_std=default_std, p_c=p_c,
                              metric=metric)]
    for i in range(epochs):
        populations.append(populations[-1].group_sex(selection_type=selection_type, save_king=save_king))
        print(f'{i}: {populations[-1].get_best_fitness()}')
    return populations


def init():
    try:
        os.mkdir('tmp')
    except FileExistsError:
        pass


def learning_curve(populations, filename='learning_curve.png', inverse=False, metric='sup'):
    if inverse:
        points = list(map(lambda n: (n[0], 1/n[1].get_best_fitness() - 1), tuple(enumerate(populations))))
    else:
        points = list(map(lambda n: (n[0], n[1].get_best_fitness()), tuple(enumerate(populations))))

    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    func = objects.PolygonalChain(points)

    settings_function = {
        'sampling rate': 30,
        'thickness': 5,
        'blur': 2,
        'color': 'gray'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }
    settings_grid = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }

    if inverse and metric == 'sup':
        frame.add_axis_surface(x_bounds=(-5, len(populations)), y_bounds=(-.55, 2.05))
        frame.blit_axes(settings_axes, x_only=False)
        frame.blit_x_grid(settings_grid, interval=len(populations) / 20, length=.014)
    else:
        frame.add_axis_surface(x_bounds=(-5, len(populations)), y_bounds=(-.55, 1.05))
        frame.blit_axes(settings_axes, x_only=False)
        frame.blit_x_grid(settings_grid, interval=len(populations) / 20, length=.007)

    frame.blit_y_grid(settings_grid, interval=.25, length=len(populations)/500)
    frame.axis_surface.blit_parametric_object(func, settings_function, interval_of_param=(0, len(populations)-1.01))
    frame.blit_axis_surface()
    frame.generate_png(filename)


def generate_curve(pop_size, unit_len, epochs, selection, std, pc, metric, inverse=False, filename=None):
    basic_func.DEBUG = True
    init()
    target = lambda x: math.sin(10 * x)
    populations_ = genetic_algorithm(target, population_size=pop_size, unit_length=unit_len, epochs=epochs,
                                     selection_type=selection, default_std=std, save_king=True, p_c=pc, metric=metric)
    if filename is None:
        learning_curve(populations_, filename=f'lc_size{pop_size}_len{unit_len}_ep{epochs}_std{std}_pc{pc}_{metric}.png'
                       , inverse=inverse, metric=metric)
    else:
        learning_curve(populations_, filename=filename, inverse=inverse, metric=metric)


if __name__ == '__main__':
    # basic_func.DEBUG = True
    # init()

    generate_curve(pop_size=50, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=100, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=200, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=350, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=500, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)

    target = lambda x: math.sin(10*x)
    populations_ = genetic_algorithm(target, population_size=100, unit_length=15, epochs=200,
                                     selection_type='rank', default_std=4, save_king=True, p_c=.1, metric='sup')
    # learning_curve(populations_, filename='lc_sup_l10.png')
    populations_ = list(map(lambda x: x.census(), populations_))
    make_film(target, populations_, filename='sup_genetic_l15.mp4', fps=10, resolution=(1280, 720), step=1, top_n=5,
              number_of_frames=80, save_ram=True, id='_gn3_', read_only=False)


