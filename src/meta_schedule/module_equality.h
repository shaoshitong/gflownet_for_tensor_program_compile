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
#ifndef TVM_META_SCHEDULE_MODULE_EQUALITY_H_
#define TVM_META_SCHEDULE_MODULE_EQUALITY_H_

#include <tvm/ir/module.h>

#include <memory>
#include <string>

namespace tvm {
namespace meta_schedule {

/*! \brief Method to compute hash and determine equality of modules  */
class ModuleEquality : public runtime::Object {
 public:
   void VisitAttrs(tvm::AttrVisitor* v) {}
  virtual ~ModuleEquality() = default;
  virtual int64_t Hash(IRModule mod) const = 0;
  virtual bool Equal(IRModule lhs, IRModule rhs) const = 0;

  /*!
   * \brief Create a ModuleEquality instance
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   *  It must be one of the followings:
   *    - "structural": Use StructuralEqual/Hash
   *    - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
   *                        equality testing and hashing.
   *    - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
   *                      given module. The "ignore-ndarray" varint is used for the extracted blocks
   *                      or in case no anchor block is found.
   *                      For the definition of the anchor block, see tvm/tir/analysis.h.
   * \return An owning pointer to the created instance
   */

  static std::unique_ptr<ModuleEquality> Create(const std::string& mod_eq_name);
  static constexpr const char* _type_key = "meta_schedule.ModuleEquality";
  TVM_DECLARE_BASE_OBJECT_INFO(ModuleEquality, Object);
};


class ModuleEqualityStructural : public ModuleEquality {
 public:
    void VisitAttrs(tvm::AttrVisitor* v) {}
  int64_t Hash(IRModule mod) const final;
  bool Equal(IRModule lhs, IRModule rhs) const final;
  static constexpr const char* _type_key = "meta_schedule.ModuleEqualityStructural";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModuleEqualityStructural, ModuleEquality);
};

class ModuleEqualityIgnoreNDArray : public ModuleEquality {
 public:
    void VisitAttrs(tvm::AttrVisitor* v) {}
  int64_t Hash(IRModule mod) const final;
  bool Equal(IRModule lhs, IRModule rhs) const final;
  static constexpr const char* _type_key = "meta_schedule.ModuleEqualityIgnoreNDArray";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModuleEqualityIgnoreNDArray, ModuleEquality);
};

class ModuleEqualityAnchorBlock : public ModuleEquality {
 public:
    void VisitAttrs(tvm::AttrVisitor* v) {}
  int64_t Hash(IRModule mod) const final;
  bool Equal(IRModule lhs, IRModule rhs) const final;
  static constexpr const char* _type_key = "meta_schedule.ModuleEqualityAnchorBlock";
  TVM_DECLARE_FINAL_OBJECT_INFO(ModuleEqualityAnchorBlock, ModuleEquality);
};

class ModuleEqualityRef: public runtime::ObjectRef {
 public:
  TVM_DLL static ModuleEqualityRef Create(const std::string& mod_eq_name);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ModuleEqualityRef, ObjectRef, ModuleEquality);
};

/*! \brief Functor to compute hash a module using the provided method. */
class ModuleHash {
 public:
  explicit ModuleHash(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}
  int64_t operator()(const IRModule& mod) const { return mod_eq_.Hash(mod); }

 private:
  const ModuleEquality& mod_eq_;
};

/*! \brief Functor to determine equality of modules using the provided method. */
class ModuleEqual {
 public:
  explicit ModuleEqual(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}
  bool operator()(const IRModule& lhs, const IRModule& rhs) const {
    return mod_eq_.Equal(lhs, rhs);
  }

 private:
  const ModuleEquality& mod_eq_;
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_MODULE_EQUALITY_H_
