#include "compiler/mlir/converters/mhlo_converter_register.h"
#include "compiler/mlir/converters/mhlo_conversion_context.h"
#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mlir_type_utils.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <torch/script.h>

namespace torch {
namespace blade {

bool ConvertAtenFakeQuantizePerTensorAffine(MhloConversionContext& ctx, const torch::jit::Node& node) {
  std::cout << "In the converter of aten fake quant!!\n";
  auto loc = GetNodeLocation(ctx, node);
  auto inp = ctx.GetMlirValue(node.input(0));
  auto scale = ctx.GetMlirValue(node.input(1));
  auto zero_point = ctx.GetMlirValue(node.input(2));
//  auto quant_min = ctx.GetMlirValue(node.input(3));
//  auto quant_max = ctx.GetMlirValue(node.input(4));

  auto builder = *ctx.builder;
  auto scaleTy = scale.getType();
  auto zpTy = zero_point.getType();
  auto tensor_scale = builder.create<mlir::tensor::FromElementsOp>(
    loc,
    mlir::RankedTensorType::get({}, scaleTy),
    scale
    );
  auto tensor_zp = builder.create<mlir::tensor::FromElementsOp>(
    loc,
    mlir::RankedTensorType::get({}, zpTy),
    zero_point
    );
//  auto scale_type = mlir::RankedTensorType::get(tensor_scale.getType().dyn_cast<mlir::RankedTensorType>().getShape(), builder.getF32Type());
//  auto zp_type = mlir::RankedTensorType::get(tensor_zp.getType().dyn_cast<mlir::RankedTensorType>().getShape(), builder.getIntegerType(32));

  auto tensor_scale_fp32 = builder.create<mlir::mhlo::ConvertOp>(loc, tensor_scale, builder.getF32Type());
  auto tensor_zp_i32 = builder.create<mlir::mhlo::ConvertOp>(loc, tensor_zp, builder.getIntegerType(32));

  auto num_bits_attr = builder.getI64IntegerAttr(8);
  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));
  auto result = builder.create<mlir::mhlo_disc::FakeQuantOp>(
    loc,
    inp.getType().dyn_cast<mlir::RankedTensorType>(),
    inp,
    tensor_scale_fp32,
    tensor_zp_i32,
    builder.getBoolAttr(true), // ise_signed
    builder.getBoolAttr(true), // use_symmetric
    builder.getI64TensorAttr(::llvm::ArrayRef<int64_t>()), // axis
    num_bits_attr, // num_bits
    builder.getI64IntegerAttr(-128), // quant_min
    builder.getI64IntegerAttr(127), // quant_max
    builder.getBoolAttr(false) // use_dynamic
    );
  ctx.value_map[node.output(0)] = result;
  return true;
}

bool ConvertAtenFakeQuantizePerChannelAffine(MhloConversionContext& ctx, const torch::jit::Node& node) {
  std::cout << "In the converter of aten fake quant!!\n";
  auto loc = GetNodeLocation(ctx, node);
  auto inp = ctx.GetMlirValue(node.input(0));
  auto scale = ctx.GetMlirValue(node.input(1));
  auto zero_point = ctx.GetMlirValue(node.input(2));
//  auto quant_min = ctx.GetMlirValue(node.input(3));
//  auto quant_max = ctx.GetMlirValue(node.input(4));

  auto builder = *ctx.builder;
  auto scaleTy = scale.getType();
  auto zpTy = zero_point.getType();
//  auto tensor_scale = builder.create<mlir::tensor::FromElementsOp>(
//    loc,
//    mlir::RankedTensorType::get({}, scaleTy),
//    scale
//    );
//  auto tensor_zp = builder.create<mlir::tensor::FromElementsOp>(
//    loc,
//    mlir::RankedTensorType::get({}, zpTy),
//    zero_point
//    );
//  auto scale_type = mlir::RankedTensorType::get(tensor_scale.getType().dyn_cast<mlir::RankedTensorType>().getShape(), builder.getF32Type());
//  auto zp_type = mlir::RankedTensorType::get(tensor_zp.getType().dyn_cast<mlir::RankedTensorType>().getShape(), builder.getIntegerType(32));

//  auto tensor_scale_fp32 = builder.create<mlir::mhlo::ConvertOp>(loc, tensor_scale, builder.getF32Type());
  auto tensor_zp_i32 = builder.create<mlir::mhlo::ConvertOp>(loc, scale, builder.getIntegerType(32));

  auto num_bits_attr = builder.getI64IntegerAttr(8);
  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));
  auto result = builder.create<mlir::mhlo_disc::FakeQuantOp>(
    loc,
    inp.getType().dyn_cast<mlir::RankedTensorType>(),
    inp,
    scale,
    tensor_zp_i32,
    builder.getBoolAttr(true), // ise_signed
    builder.getBoolAttr(true), // use_symmetric
    builder.getI64TensorAttr(::llvm::ArrayRef<int64_t>({1,})), // axis
    num_bits_attr, // num_bits
    builder.getI64IntegerAttr(-128), // quant_min
    builder.getI64IntegerAttr(127), // quant_max
    builder.getBoolAttr(false) // use_dynamic
    );
  ctx.value_map[node.output(0)] = result;
  return true;
}


namespace {
auto mhlo_conversion = 
  MhloConversionPatternRegister()
    .pattern(R"SIG(aten::fake_quantize_per_tensor_affine(
                   Tensor input, float scale, int zero_point,
                   int quant_min, int quant_max) -> Tensor)SIG",
            ConvertAtenFakeQuantizePerTensorAffine)
    .pattern(R"SIG(aten::fake_quantize_per_channel_affine(
                   Tensor input, float scale, int zero_point, int axis,
                   int quant_min, int quant_max) -> Tensor)SIG",
            ConvertAtenFakeQuantizePerChannelAffine);
}
} // namespace blade
} // namespace torch