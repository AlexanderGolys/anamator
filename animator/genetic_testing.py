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
    generate_curve(pop_size=50, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=100, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=200, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=350, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    generate_curve(pop_size=500, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)

    target = lambda x: math.sin(10*x)
    populations_ = genetic_algorithm(target, population_size=500, unit_length=15, epochs=500,
                                     selection_type='rank', default_std=8, save_king=True, p_c=.15, metric='int')
    learning_curve(populations_, filename='lc_int_op.png')
    populations_ = list(map(lambda x: x.census(), populations_))
    make_film(target, populations_, filename='int_genetic_op.mp4', fps=15, resolution=(1280, 720), step=1, top_n=5,
              number_of_frames=250, save_ram=True, id='_gnop_', read_only=False)
