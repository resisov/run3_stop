"""Preserve CMSSW Python initialization while enforcing scratch-only I/O."""
import os
import sys


def scratch_guard(event, args):
    if event in ('open', 'os.chdir', 'os.listdir', 'os.scandir') and args:
        value = args[0]
        if isinstance(value, (str, bytes)):
            path = os.path.abspath(os.fsdecode(value))
            if path in ('/home', '/data') or path.startswith(('/home/', '/data/')):
                raise PermissionError('Prohibited storage path: ' + path)


sys.addaudithook(scratch_guard)
if 'PYTHON3PATH' in os.environ:
    import site
    for path in os.environ['PYTHON3PATH'].split(os.pathsep):
        if not path.startswith(('/cvmfs/', '/hep2-scratch/')):
            raise PermissionError('Unexpected CMSSW Python path: ' + path)
        site.addsitedir(path)
