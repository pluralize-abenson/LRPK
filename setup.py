from setuptools import setup, find_packages

CLASSIFIERS = """
Development Status :: 1 - Planning
Intended Audience :: Other Audience
Intended Audience :: Developers
License :: OSI Approved :: BSD License

Programming Language :: Python :: 3.6
Topic :: Software Development
Topic :: Artistic Software
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

setup(
    name='RPY',
    version=' 0.1',
    author='Alexander Benson',
    author_email='alexander.benson@stmartin.edu',
    url='',
    description='Tools for managing robot paths',
    #long_description=long_description,
    packages=find_packages(),
    classifiers=[f for f in CLASSIFIERS.split('\n') if f],
    install_requires=['numpy',
                      'matplotlib'],
)