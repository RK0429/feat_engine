from setuptools import setup, find_packages

setup(
    name='feat_engine',                                             # Package name
    version='1.0.0',                                                # Version
    description='A comprehensive feature engineering toolkit.',     # Short description
    long_description=open('README.md').read(),                      # Detailed description from README.md
    long_description_content_type='text/markdown',                  # Description content type
    author='Ryota Kobayashi',                                       # Your name
    author_email='s.woods.m.29@gmail.com',                          # Your email
    url='https://github.com/RK0429/feat_engine',                    # Optional: Link to GitHub repository
    license='MIT',                                                  # License type
    packages=find_packages(),                                       # Automatically find all packages and modules
    install_requires=[                                              # Package dependencies
        'numpy',
        'pandas',
        'scikit-learn',
    ],
    classifiers=[                                                   # Optional classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',                                        # Minimum Python version required
)
