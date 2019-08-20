from setuptools import setup, find_packages
import sys, os


from setuptools.command.install import install as DistutilsInstall
from setuptools.command.egg_info import egg_info as EggInfo

setup_py_dir = os.path.dirname(os.path.realpath(__file__))


need_files = ['cpp_household.so']
hh = setup_py_dir + "/roboschoolfork_nao"

for root, dirs, files in os.walk(hh):
    for fn in files:
        ext = os.path.splitext(fn)[1][1:]
        if ext and ext in 'png jpg urdf obj mtl dae off stl STL xml glsl so 87 dylib'.split():
            fn = root + "/" + fn
            need_files.append(fn[1+len(hh):])
for x in need_files: print(x)
print("found resource files: %i" % len(need_files))
#for n in need_files: print("-- %s" % n)

setup(
    name = 'roboschoolfork_nao',
    version = '1.0',
    description = 'OpenAI Household Simulator: mobile manipulation using Bullet',
    maintainer = 'Oleg Klimov',
    maintainer_email = 'omgtech@gmail.com',
    url = 'https://github.com/openai/roboschool',
    packages=[x for x in find_packages()],
    package_data = { '': need_files }
    )
