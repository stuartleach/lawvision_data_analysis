from setuptools import setup, find_packages


# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]


# Read the requirements from the requirements.txt file
requirements = parse_requirements('requirements.txt')

setup(
    name='lawvision_data_analysis',
    version='0.1.0',
    description='Utility functions for the bail project.',
    author='J. Stuart Leach, Jr.',
    author_email='jsleach@proton.me',
    url='https://github.com/stuartleach/lawvision_data_analysis',  # Replace with your actual GitHub repo URL
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: GNU for non-commercial use',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
)
