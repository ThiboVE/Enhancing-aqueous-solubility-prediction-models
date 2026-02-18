from setuptools import setup, find_packages

setup(
    name='library', # Import name for your package
    version='0.0.1',
    packages=find_packages(where='.'),
    package_dir={"": "."},  
)