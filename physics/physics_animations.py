from physics_main import *


def first_animation(filename, duration):
    film = Film(24, FHD, 'first')

    engine = ElasticGravity(gravity_acc=(0, -.25))
    env = Environment()
    b1 = BallParticle(env, radius=50, init_position=(0, 40), init_velocity=(5, 10), init_acceleration=(0, 0))
    b2 = BallParticle(env, radius=50, init_position=(200, 0), init_velocity=(-10, 20), init_acceleration=(0, 0))
    env.add_objs(b1, b2)

    for i in range(duration):
        env.tick()
        frame = OneAxisFrame(FHD, x_padding=0, y_padding=0)
        bounds = [(-960, 960), (-520, 520)]
        frame.add_axis_surface(*bounds)
        for obj in env.objects:
            ball = obj.create_object()
            frame.axis_surface.blit_bitmap_object(obj.pos, ball, BitmapBlittingSettings())
            print(obj)
        frame.blit_axis_surface()
        film.add_frame(frame, True)

    film.render(filename, True)


if __name__ == '__main__':
    first_animation('first.mp4', 72)

