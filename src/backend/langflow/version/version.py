from importlib import metadata
is_pre_release = False
try:
    __version__ = metadata.version('ragstack-ai-langflow')
except metadata.PackageNotFoundError:
    __version__ = ''
del metadata

