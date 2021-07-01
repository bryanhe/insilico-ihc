import os
import configparser
import types

FILENAME = None
param = {}
for filename in ["dstain.cfg",
                 ".dstain.cfg",
                 os.path.expanduser("~/dstain.cfg"),
                 os.path.expanduser("~/.dstain.cfg")]:
    # TODO: default when installing
    if os.path.isfile(filename):
        FILENAME = filename
        config = configparser.ConfigParser()
        with open(filename, "r") as f:
            config.read_string("[config]\n" + f.read())
            param = config["config"]
        break

config = types.SimpleNamespace(
    ROOT=os.path.dirname(__file__),
    FILENAME=FILENAME,
    DATA_RAW=os.path.expanduser(param.get("data_raw", os.path.join("data", "raw"))),
    REGISTRATION=os.path.expanduser(param.get("registration", os.path.join("data", "registration"))),
    ANNOTATION=os.path.expanduser(param.get("annotation", os.path.join("data", "annotations"))),
    OUTPUT=os.path.expanduser(param.get("output", "output")),
    CACHE=os.path.expanduser(param.get("cache", ".cache")),
)
