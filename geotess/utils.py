from collections import namedtuple

# These namedtuples are lightweight readable containers for Layer/Attribute info
Layer = namedtuple('Layer', ['name', 'tess_id'])
Attribute = namedtuple('Attribute', ['name', 'unit'])
