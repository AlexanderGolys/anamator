import functools

from abc import abstractmethod, ABC

from src.anamator.basic_func import *
from src.anamator.objects import *


class PhysicsEngine(ABC):
    @staticmethod
    @abstractmethod
    def collision(obj1, obj2):
        pass

    @abstractmethod
    def time_passes(self, obj):
        pass


class ElasticGravity(PhysicsEngine):
    def __init__(self, gravity_acc=(0, -1)):
        self.g = gravity_acc

    @staticmethod
    def collision(obj1, obj2):
        obj1.v *= -1
        obj1.a *= -1
        obj2.v *= -1
        obj2.a *= -1

    def time_passes(self, obj):
        obj.v += self.g


class Environment:
    def __init__(self, *objs):
        self.objects = objs

    def add_objs(self, *objs):
        self.objects += objs

    def tick(self):
        for obj in self.objects:
            obj.update_state()


class Thing(ABC):
    @abstractmethod
    def __init__(self, environment, engine=ElasticGravity()):
        self.env = environment
        self.engine = engine
        self.hash = np.random.randint(0, 10000000000)

    @abstractmethod
    def __sub__(self, other):
        pass

    def __eq__(self, other):
        return self.hash == other.hash

    @abstractmethod
    def update_state(self):
        pass


class BallParticle(Thing):
    def __init__(self, environment, radius=50, init_position=(0, 0), init_velocity=(0, 0), init_acceleration=(1, 1),
                 engine=ElasticGravity(), radius_px=None, parametric=False, color='white', opacity=1):
        super().__init__(environment, engine)
        self.pos = np.array(init_position)
        self.v = np.array(init_velocity)
        self.a = np.array(init_acceleration)
        self.r = radius
        self.radius_px = radius_px if radius_px is not None else radius
        self.parametric = parametric
        self.color = color
        self.opacity = opacity

    @functools.singledispatchmethod
    def __sub__(self, other):
        if isinstance(other, Wall):
            x, y = self.pos
            return abs(other.a*x + other.b*y + other.c)/np.linalg.norm([other.a, other.b])
        return np.linalg.norm(self.pos - other.pos) - self.r - other.r

    def move(self):
        self.pos += self.v
        self.v += self.a

    def update_state(self):
        self.engine.time_passes(self)
        for obj in self.env.objects:
            if self.detect_collision(obj) and obj != self:
                # print('bum')
                self.engine.collision(self, obj)
        self.move()

    def detect_collision(self, other, threshold=2):
        print(self - other)
        return self - other < threshold

    def create_parametric_object(self):
        return Disk(self.pos, self.r)

    def create_bitmap_object(self):
        return BitmapDisk(self.radius_px, self.color, self.opacity)

    def create_object(self):
        if self.parametric:
            return self.create_parametric_object()
        return self.create_bitmap_object()

    def __str__(self):
        return f'Ball(p={self.pos}, v={self.v}, a={self.a})'


class Wall(Thing):
    def __init__(self, environment, a, b, c, engine=ElasticGravity()):
        super().__init__(environment, engine)
        self.a = a
        self.b = b
        self.c = c

    @functools.singledispatchmethod
    def __sub__(self, other):
        return other - self

    def update_state(self):
        pass


if __name__ == '__main__':
    pass


