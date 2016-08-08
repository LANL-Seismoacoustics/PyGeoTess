from geotess import Layer, Attribute, Model
from geotess.libgeotess import AK135Model

descr = """
Simple example of populating a 3D GeoTess model
comprised of 3 multi-level tessellations
author: Sandy Ballard
contact: sballar@sandia.gov"""

# Layer and Attribute are named tuples, which means they are two-tuples of 
# (name, tess_id) or (name, unit), the contents of which can be accessed 
# using names, like: inner_core.tess_id or rho.name
layers = (Layer(name='inner core', tess_id=0),
          Layer('outer core', 0),
          Layer('lower mantle', 1),
          Layer('lower mantle', 1),
          Layer('transition zone', 1),
          Layer('upper mantle', 1),
          Layer('lower crust', 2),
          Layer('upper crust', 2))

attributes = [Attribute(name='Vp', unit='km/sec'),
              Attribute('Vs', 'km/sec'),
              Attribute('rho', 'g/cc')]

# Initialize the model.  It is full of null data.
model = Model('geotess/data/small_model_grid.ascii', layers, attributes,
              'float', description=descr)

ak135 = AK135Model()

# populate the model from ak135
for layer_num in range(len(model.layers)):
    for vert_num, (lat, lon) in enumerate(model.grid.vertices()):
        radii, attrib_values = ak135.getLayerProfile(lat, lon, layer_num)
        model.set_profile(vert_num, layer_num, radii, attrib_values)


# write to file
model.write('path/to/output/small_model.ascii')


