// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pybind_functions.h"
#include "placeholder_op.h"

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/script.h>

namespace torch {
namespace blade {
namespace quantization {
using namespace torch::jit;

void add_placeholder_for_fake_quant(Module& model) {
  Symbol sym = Symbol::fromQualString(
      torch::blade::quantization::custom_placeholder_name);
  auto g = model.get_method("forward").graph();
  // the graph should be inlined first
  Inline(*g);
  // add a placeholder op after each aten::fake_quantize_per_channel_affine
  // node, which is to prevent the aten::fake_quantize_per_channel_affine be
  // folded by the ConstantPropagation pass.
  for (auto n : g->nodes()) {
    if (n->kind().toQualString() ==
        std::string(torch::blade::quantization::
                        at_fake_quant_per_channel_affine_name)) {
      auto place_holder = g->appendNode(g->create(sym));
      place_holder->moveAfter(n);
      n->outputs()[0]->replaceAllUsesWith(place_holder->outputs()[0]);
      place_holder->addInput(n->outputs()[0]);
    }
  }
}

void remove_placeholder(Module& model) {
  auto g = model.get_method("forward").graph();
  std::vector<Node*> place_holder_nodes;
  for (auto n : g->nodes()) {
    if (n->kind().toQualString() ==
        std::string(torch::blade::quantization::custom_placeholder_name)) {
      n->outputs()[0]->replaceAllUsesWith(n->inputs()[0]);
      n->removeAllInputs();
      place_holder_nodes.push_back(n);
    }
  }
  for (auto n : place_holder_nodes) {
    n->destroy();
  }
}

//  %weight.2 = torch_blade_quantization::placeholder(%weight.1)
//  %weight.3 = aten::t(%weight.2)
//  %y = aten::matmul(%x, %weight.3)
// fake_quantize_per_channel_affine

void fold_transpose(Module& model, Graph& pattern){
  auto g = model.get_method("forward").graph();
  std::cout << "pattern graph: " << pattern.toString() << std::endl;
  auto matched = findPatternMatches(pattern, *g);
  std::cout << "Find match nums: " << matched.size() << std::endl;

}

void initQuantizationBindings(py::module& m) {
  py::module quantization = m.def_submodule(
      "_quantization", "torch_blade python bindings for quantization");
  quantization.def(
      "add_placeholder_for_fake_quant", &add_placeholder_for_fake_quant);
  quantization.def("remove_placeholder", &remove_placeholder);
  quantization.def("fold_transpose", &fold_transpose);
  quantization.attr("at_fake_quant_per_tensor_affine_name") =
      torch::blade::quantization::at_fake_quant_per_tensor_affine_name;
  quantization.attr("at_fake_quant_per_channel_affine_name") =
      torch::blade::quantization::at_fake_quant_per_channel_affine_name;
}

} // namespace quantization
} // namespace blade
} // namespace torch
