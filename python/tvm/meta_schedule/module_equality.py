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
    """base"""
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ModuleEquality, ) # type: ignore # pylint: disable=no-member
        
    def hash(self, mod) -> int:
        raise NotImplementedError
    
    def equal(self, lhs, rhs) -> bool:
        raise NotImplementedError
    
@register_object("meta_schedule.ModuleEqualityStructural")
class ModuleEqualityStructural(Object):
    """structural"""
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ModuleEqualityStructural,  # type: ignore # pylint: disable=no-member
        )
    def hash(self, mod) -> int:
        return _ffi_api.ModuleEqualityStructuralHash(self, mod)
    def equal(self, lhs, rhs) -> bool:
        return _ffi_api.ModuleEqualityStructuralEqual(self, lhs, rhs)

@register_object("meta_schedule.ModuleEqualityIgnoreNDArray")
class ModuleEqualityIgnoreNDArray(Object):
    """ignore-ndarray"""
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ModuleEqualityIgnoreNDArray,  # type: ignore # pylint: disable=no-member
        )
    def hash(self, mod) -> int:
        return _ffi_api.ModuleEqualityIgnoreNDArrayHash(self, mod)
    def equal(self, lhs, rhs) -> bool:
        return _ffi_api.ModuleEqualityIgnoreNDArrayEqual(self, lhs, rhs)

@register_object("meta_schedule.ModuleEqualityAnchorBlock")
class ModuleEqualityAnchorBlock(Object):
    """anchor-block"""
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ModuleEqualityAnchorBlock,  # type: ignore # pylint: disable=no-member
        )
    def hash(self, mod) -> int:
        return _ffi_api.ModuleEqualityAnchorBlockHash(self, mod)
    def equal(self, lhs, rhs) -> bool:
        return _ffi_api.ModuleEqualityAnchorBlockEqual(self, lhs, rhs)