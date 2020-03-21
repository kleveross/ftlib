import logging
import os
import signal
import time

import numpy as np

from ftlib.commlib.basic_commlib import BasicCommLib
from ftlib.commlib.gloo import gloo_lib  # type: ignore