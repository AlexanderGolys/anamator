from genetic import *


if __name__ == '__main__':
    # basic_func.DEBUG = True
    # init()
    # target = lambda x: math.sin(10*x)
    # populations_ = genetic_algorithm(target, population_size=40, unit_length=10, epochs=120,
    #                                  selection_type='rank', default_std=1, save_king=True, p_c=.4, metric='int')
    # print('dupa')
    # learning_curve(populations_, filename=f'learning_curve_rank_3.png')
    # populations_ = list(map(lambda x: x.census(), populations_))
    # make_film(target, populations_, filename='genetic_diversity_2.mp4', fps=1, resolution=(1280, 720), step=1, top_n=5,
    #           number_of_frames=8, save_ram=True, id='_gndiv_', read_only=False)
    # generate_curve(pop_size=100, unit_len=15, epochs=200, selection='rank', std=4, pc=.1, metric='int', inverse=True,
    #                filename='test.png')

    target = lambda x: 5*math.sin(10*x)*math.exp(x/3)
    # target = lambda x: math.sin(10 * x)

    populations_ = genetic_algorithm(target, population_size=100, unit_length=21, epochs=1000,
                                      selection_type='rank', default_std=5, save_king=True, p_c=.65, metric='int')
    learning_curve(populations_, filename='symmetric_big_21.png', inverse=True)
    populations_ = list(map(lambda x: x.census(), populations_))
    make_film(target, populations_, filename='symmetric_big_21.mp4', fps=5, resolution=(1280, 720), step=1, top_n=5,
              number_of_frames=60, save_ram=True, id='_sym_slow_', read_only=False)