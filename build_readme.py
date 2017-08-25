def _read_readme():

    with open('docs/readme.rst', 'r') as readme:
        return readme.read()


def _substitute(readme):

    readme = (readme.replace('_static', 'docs/_static')
              .replace('.. testcode::', '.. code-block:: python')
              .replace('.. testoutput::\n   :hide:', ''))

    return readme


def _write(readme):

    with open('readme.rst', 'w') as out:
        out.write(readme)


if __name__ == '__main__':

    readme = _read_readme()
    readme = _substitute(readme)
    _write(readme)
