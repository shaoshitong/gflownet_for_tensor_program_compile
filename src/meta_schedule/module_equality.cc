/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "module_equality.h"

#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/tir/analysis.h>

#include <memory>

#include "../node/ndarray_hash_equal.h"

namespace tvm {
namespace meta_schedule {

int64_t ModuleEqualityStructural::Hash(IRModule mod) const { return tvm::StructuralHash()(mod); }

bool ModuleEqualityStructural::Equal(IRModule lhs, IRModule rhs) const { return tvm::StructuralEqual()(lhs, rhs); }


// class ModuleEqualityStructuralRef: public ModuleEqualityRef {
//   public:
//     using ModuleEqualityRef::ModuleEqualityRef;
// };

class SEqualHandlerIgnoreNDArray : public SEqualHandlerDefault {
 public:
  SEqualHandlerIgnoreNDArray() : SEqualHandlerDefault(false, nullptr, false) {}

 protected:
  bool DispatchSEqualReduce(const ObjectRef& lhs, const ObjectRef& rhs, bool map_free_vars,
                            const Optional<ObjectPathPair>& current_paths) {
    if (auto lhs_ptr = lhs.as<runtime::NDArray::Container>(),
        rhs_ptr = rhs.as<runtime::NDArray::Container>();
        lhs_ptr && rhs_ptr) {
      SEqualReducer reducer(this, nullptr, map_free_vars);
      return NDArrayEqual(lhs_ptr, rhs_ptr, reducer, false);
    }
    return SEqualHandlerDefault::DispatchSEqualReduce(lhs, rhs, map_free_vars, current_paths);
  }
};

class SHashHandlerIgnoreNDArray : public SHashHandlerDefault {
 protected:
  void DispatchSHash(const ObjectRef& object, bool map_free_vars) override {
    ICHECK(object.defined());
    if (auto ndarray = object.as<runtime::NDArray::Container>()) {
      SHashReducer hash_reduce(this, map_free_vars);
      NDArrayHash(ndarray, &hash_reduce, false);
    } else {
      SHashHandlerDefault::DispatchSHash(object, map_free_vars);
    }
  }
};


int64_t ModuleEqualityIgnoreNDArray::Hash(IRModule mod) const { return SHashHandlerIgnoreNDArray().Hash(mod, false); }

bool ModuleEqualityIgnoreNDArray::Equal(IRModule lhs, IRModule rhs) const {
    return SEqualHandlerIgnoreNDArray().Equal(lhs, rhs, false);
  }

// class ModuleEqualityIgnoreNDArrayRef : public ModuleEqualityRef {
//   public:
//     using ModuleEqualityRef::ModuleEqualityRef;
// };

// The NDArray-ignoring variant of structural equal / hash is used for the module equality
// on the extracted anchor blocks.
int64_t ModuleEqualityAnchorBlock::Hash(IRModule mod) const {
  auto anchor_block = tir::FindAnchorBlock(mod);
  if (anchor_block) {
    return SHashHandlerIgnoreNDArray().Hash(GetRef<tir::Block>(anchor_block), false);
  }
  return ModuleEqualityIgnoreNDArray().Hash(mod);
}
bool ModuleEqualityAnchorBlock::Equal(IRModule lhs, IRModule rhs) const {
  auto anchor_block_lhs = tir::FindAnchorBlock(lhs);
  auto anchor_block_rhs = tir::FindAnchorBlock(rhs);
  if (anchor_block_lhs && anchor_block_rhs) {
    return SEqualHandlerIgnoreNDArray().Equal(GetRef<tir::Block>(anchor_block_lhs),
                                              GetRef<tir::Block>(anchor_block_rhs), false);
  }
  return ModuleEqualityIgnoreNDArray().Equal(lhs, rhs);
}

// class ModuleEqualityAnchorBlockRef : public ModuleEqualityRef {
//   public:
//     using ModuleEqualityRef::ModuleEqualityRef;
// };

std::unique_ptr<ModuleEquality> ModuleEquality::Create(const std::string& mod_eq_name) {
  if (mod_eq_name == "structural") {
    return std::make_unique<ModuleEqualityStructural>();
  } else if (mod_eq_name == "ignore-ndarray") {
    return std::make_unique<ModuleEqualityIgnoreNDArray>();
  } else if (mod_eq_name == "anchor-block") {
    return std::make_unique<ModuleEqualityAnchorBlock>();
  }
  LOG(FATAL) << "Unknown module equality " << mod_eq_name;
}

ModuleEqualityRef ModuleEqualityRef::Create(const std::string& mod_eq_name) {
  if (mod_eq_name == "structural") {
    ObjectPtr<ModuleEqualityStructural> n = make_object<ModuleEqualityStructural>();
    return ModuleEqualityRef(n);
} else if (mod_eq_name == "ignore-ndarray") {
    ObjectPtr<ModuleEqualityIgnoreNDArray> n = make_object<ModuleEqualityIgnoreNDArray>();
    return ModuleEqualityRef(n);
  } else if (mod_eq_name == "anchor-block") {
    ObjectPtr<ModuleEqualityAnchorBlock> n = make_object<ModuleEqualityAnchorBlock>();
    return ModuleEqualityRef(n);
  }
    LOG(FATAL) << "Unknown module equality " << mod_eq_name;
}

TVM_REGISTER_OBJECT_TYPE(ModuleEquality);
TVM_REGISTER_NODE_TYPE(ModuleEqualityStructural);
TVM_REGISTER_NODE_TYPE(ModuleEqualityIgnoreNDArray);
TVM_REGISTER_NODE_TYPE(ModuleEqualityAnchorBlock);

// TVM_REGISTER_GLOBAL("meta_schedule.ModuleEquality").set_body_typed([]() -> ModuleEqualityRef {
//   return ModuleEqualityStructuralRef();
// });
// TVM_REGISTER_GLOBAL("meta_schedule.ModuleEqualityIgnoreNDArray").set_body_typed([]() -> ModuleEqualityIgnoreNDArrayRef {
//   return ModuleEqualityIgnoreNDArrayRef();
// });
// TVM_REGISTER_GLOBAL("meta_schedule.ModuleEqualityAnchorBlock").set_body_typed([]() -> ModuleEqualityAnchorBlockRef {
//   return ModuleEqualityAnchorBlockRef();
// });
TVM_REGISTER_GLOBAL("meta_schedule.ModuleEqualityCreate").set_body_typed(ModuleEqualityRef::Create);
TVM_REGISTER_GLOBAL("meta_schedule.ModuleEqualityHash")
    .set_body_method<ModuleEqualityRef>(&ModuleEquality::Hash);
TVM_REGISTER_GLOBAL("meta_schedule.ModuleEqualityEqual")
    .set_body_method<ModuleEqualityRef>(&ModuleEquality::Equal);

}  // namespace meta_schedule
}  // namespace tvm
