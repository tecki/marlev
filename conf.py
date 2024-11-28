import sys

sys.path.insert(0, __file__.rsplit('/', 1)[0])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
numfig = True
source_suffix = '.rst'
master_doc = 'index'

project = 'marlev'
copyright = '2024, European XFEL GmbH'
author = 'Martin Teichmann'

release = "0.1"
version = "0.1.0"
language = "en"
exclude_patterns = ['build']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'alabaster'
html_static_path = ['build/static']
htmlhelp_basename = 'marlev'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    }
