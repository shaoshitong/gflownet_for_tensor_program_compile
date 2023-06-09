# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""A context manager that profiles tuning time cost for different parts."""
from contextlib import contextmanager
from typing import Dict, Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api

@register_object("meta_schedule.ModuleEquality")
class ModuleEquality(Object):
    """
    A TVM object cost model to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyCostModel
    """

    def __init__(
        self,
        mod_eq_name
    ):
        self.mod_eq_name = mod_eq_name
        self.__init_handle_by_constructor__(
            _ffi_api.ModuleEqualityCreate,  # type: ignore # pylint: disable=no-member
            mod_eq_name
        )
    def hash(self, mod) -> int:
        return _ffi_api.ModuleEqualityHash(self, mod)
    
    def equal(self, lhs, rhs) -> bool:
        return _ffi_api.ModuleEqualityEqual(self, lhs, rhs)