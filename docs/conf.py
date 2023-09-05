import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

import simple_transforms

version = simple_transforms.__version__
release = simple_transforms.__version__

project = 'simple_transforms'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx-prompt',
]

todo_include_todos = True

source_suffix = {
    '.rst':'restructuredtext',
    '.md':'markdown',
}

master_doc = 'index'
language = 'en'

html_theme = 'furo'

add_module_names = False
autodoc_typehints = 'none'
