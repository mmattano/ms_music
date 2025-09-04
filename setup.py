from setuptools import setup, find_packages


install_requires = [
    # Core dependencies
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'scipy>=1.7.0',
    'matplotlib>=3.3.0',
    'tqdm>=4.60.0',

    # Mass spectrometry data support
    'matchms>=0.25.0',

    # Audio processing
    'librosa>=0.9.0',

    # MIDI support
    'mido>=1.2.10',

    # Visualizations
    'seaborn>=0.11.0',
]

setup(
    name='ms_music',
    version='0.2.0',
    author='Matthias Anagho-Mattanovich',
    author_email='matthias.mattanovich@sund.ku.dk',
    description='A Python package to generate audio and MIDI'
    ' from mass spectrometry data.',
    long_description='A Python package to generate audio and MIDI'
    ' from mass spectrometry data.',
    long_description_content_type='text/markdown',
    url='https://github.com/mmattano/ms_music',
    packages=find_packages(),
    python_requires='>=3.8',

    install_requires=install_requires,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Artistic Software',
    ],

    keywords=[
        'mass-spectrometry', 'sonification', 'audio', 'midi', 'bioinformatics',
        'data-visualization', 'microtonal', 'music-theory',
        'equal-temperament',
        'just-intonation', 'experimental-music', 'scientific-data',
        'quantization'
    ],

    project_urls={
        'Bug Reports': 'https://github.com/mmattano/ms_music/issues',
        'Source': 'https://github.com/mmattano/ms_music',
        # 'Documentation': 'https://ms-music.readthedocs.io/',
        'Examples': 'https://github.com/mmattano/ms_music/tree/main/examples',
    },

    include_package_data=True,
    zip_safe=False,
)
