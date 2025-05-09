#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для расширенной оптимизации и валидации моделей классификации лояльности клиентов.
"""

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import numpy as np
import pandas as pd 