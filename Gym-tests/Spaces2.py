#This introspection can be helpful to write generic code that works for many different environments. Box and Discrete are the most common Spaces. You can sample from a Space or check that something belongs to it:

from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8