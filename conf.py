extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
numfig = True
source_suffix = '.rst'
master_doc = 'index'

project = 'levmarpy'
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
htmlhelp_basename = 'levmarpy'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    }
