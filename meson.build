project('fast_g3', 'cpp',  default_options : ['cpp_std=c++17'])
py = import('python').find_installation(pure: false)
incdir_numpy = run_command( py, ['-c', 'import numpy; import os; print(os.path.relpath(numpy.get_include()))'], check: true).stdout().strip()
flac  = dependency('flac')
bzip2 = dependency('bzip2')
openmp= dependency('openmp')

py.install_sources(
  ['fast_g3/__init__.py', 'fast_g3/g3file.py'],
  subdir: 'fast_g3',
)

py.extension_module(
  'fast_g3_core',
  'src/fast_g3_core.cpp',
  include_directories: [incdir_numpy],
  dependencies: [flac, bzip2, openmp],
  cpp_args: ['-Wfatal-errors'],
  subdir: 'fast_g3',
  install: true,
)
