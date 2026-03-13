# -*- coding: utf-8 -*-
import os
import sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append('/home/bioinfo/pycharmproject_caojie38/evaluation/mapping_optim_v1')
class CamError(Exception):
    pass


class PathNotFoundError(CamError):
    pass


class InvalidMappingToolError(CamError):
    pass


class CamFileAccessModeError(CamError):
    pass


class GetRefFileError(CamError):
    pass