
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include <iostream>
#define DEBUG_TYPE disc-fold-const-for-fake-quant-op

namespace mlir {
namespace disc_ral {
namespace {

struct FakeQuantDotGeneralDynamicReshapeConverter2
    : public OpRewritePattern<mhlo_disc::FakeQuantOp> {
  using OpRewritePattern<mhlo_disc::FakeQuantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo_disc::FakeQuantOp op,
                                PatternRewriter& rewriter) const override {
    std::cout << "match start2222!!!!\n";
    auto output_dynamic_reshape_op = op.input().template getDefiningOp<mhlo::DynamicReshapeOp>();
    if (!output_dynamic_reshape_op) {
        std::cout << "Not dynamic reshape OP for output\n";
        return failure();
    }
    auto dot_general_op = output_dynamic_reshape_op.operand().template getDefiningOp<mhlo::DotGeneralOp>();
    if (!dot_general_op) {
        std::cout << "Not dot general OP\n";
        return failure();
    }

    auto input_dynamic_reshape_op = dot_general_op.lhs().template getDefiningOp<mhlo::DynamicReshapeOp>();
    if (!input_dynamic_reshape_op) {
        std::cout << "Not dynamic reshape OP for input!\n";
        return failure();
    }

    auto input_fake_quant_op = input_dynamic_reshape_op.operand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!input_fake_quant_op) {
        std::cout << "Not fake quant OP for input!\n";
        return failure();
    }

    auto weight_dynamic_reshape = dot_general_op.rhs().template getDefiningOp<mhlo::DynamicReshapeOp>();
    if (!weight_dynamic_reshape) {
        std::cout << "Not dynamic reshape OP for weight!\n";
        return failure();
    }

    auto weight_transpose_op = weight_dynamic_reshape.operand().template getDefiningOp<mhlo::TransposeOp>();
    if (!weight_transpose_op) {
        std::cout << "not transpose op for the weight\n";
        return failure();
    }

    auto weight_fake_quant_op = weight_transpose_op.getOperand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!weight_fake_quant_op) {
        std::cout << "no fake quant op for weight\n";
        return failure();
    }

    auto const_op = weight_fake_quant_op.input().template getDefiningOp<mhlo::ConstantOp>();
    if (!const_op) {
        std::cout << "Not constant OP for weight!\n";
        return failure();
    }
    std::cout << "match success\n";

    auto dynamic_reshape_ty = input_dynamic_reshape_op.getType().cast<RankedTensorType>();
    auto dynamic_reshape_out_shape =input_dynamic_reshape_op.output_shape();
    auto new_input_dynamic_reshape_output = rewriter.create<mhlo::DynamicReshapeOp>(
          op->getLoc(),
          dynamic_reshape_ty,
          input_fake_quant_op.input(), dynamic_reshape_out_shape);
    auto new_input_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_input_dynamic_reshape_output.getType(),
        new_input_dynamic_reshape_output,
        input_fake_quant_op.scale(),
        input_fake_quant_op.zero_point(),
        input_fake_quant_op.use_signedAttr(),
        input_fake_quant_op.use_symmetricAttr(),
        input_fake_quant_op.axisAttr(),
        input_fake_quant_op.num_bitsAttr(),
        input_fake_quant_op.quant_minAttr(),
        input_fake_quant_op.quant_maxAttr(),
        input_fake_quant_op.use_dynamicAttr(),
        input_fake_quant_op.round_modeAttr()
    );

    auto new_weight_transpose_op = rewriter.create<mhlo::TransposeOp>(
        op->getLoc(),
        weight_transpose_op.getType(),
        const_op,
        weight_transpose_op.permutationAttr()
    );

    auto weight_dynamic_reshape_ty = weight_dynamic_reshape.getType().cast<RankedTensorType>();
    auto weight_dynamic_reshape_shape = weight_dynamic_reshape.output_shape();
    auto new_weight_dynamic_reshape_output = rewriter.create<mhlo::DynamicReshapeOp>(
          op->getLoc(),
          weight_dynamic_reshape_ty,
          new_weight_transpose_op, weight_dynamic_reshape_shape);

    auto permutation_iter = weight_transpose_op.permutationAttr().getValues<int64_t>();
    std::vector<int64_t> permutation_vec;
    for (auto p: permutation_iter) {
        permutation_vec.push_back(p);
    }
    auto origin_axis_iter = weight_fake_quant_op.axisAttr().getValues<int64_t>();
    std::vector<int64_t> origin_axis_vec;
    for (auto a: origin_axis_iter) {
        origin_axis_vec.push_back(a);
    }
    int64_t new_axis;
    for (int64_t i=0; i<permutation_vec.size(); i++) {
        if(permutation_vec[i] == origin_axis_vec[0]) {
            new_axis = i;
            break;
        }
    }
    std::cout << "new_axis: " << new_axis << std::endl;
    auto attrTy = RankedTensorType::get({static_cast<int64_t>(1)},
                                      rewriter.getIntegerType(64));
    auto permuteAttr = DenseIntElementsAttr::get(attrTy, {new_axis});
    auto new_weight_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_weight_dynamic_reshape_output.getType(),
        new_weight_dynamic_reshape_output,
        weight_fake_quant_op.scale(),
        weight_fake_quant_op.zero_point(),
        weight_fake_quant_op.use_signedAttr(),
        weight_fake_quant_op.use_symmetricAttr(),
        permuteAttr,
        weight_fake_quant_op.num_bitsAttr(),
        weight_fake_quant_op.quant_minAttr(),
        weight_fake_quant_op.quant_maxAttr(),
        weight_fake_quant_op.use_dynamicAttr(),
        weight_fake_quant_op.round_modeAttr()
    );
    auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
      rewriter.getContext(), {}, {},
      {1}, {0});

    auto new_dot_op = rewriter.create<mhlo::DotGeneralOp>(
        op->getLoc(), dot_general_op.getType(),
        new_input_fake_quant, new_weight_fake_quant, dot_dimension_attr, nullptr
    );
//    auto new_dot_op = rewriter.create<mhlo::DotOp>(
//        op->getLoc(), new_input_fake_quant, new_weight_fake_quant, nullptr
//    );
    auto new_output_fake_quant =  rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_dot_op.getType(),
        new_dot_op,
        op.scale(),
        op.zero_point(),
        op.use_signedAttr(),
        op.use_symmetricAttr(),
        op.axisAttr(),
        op.num_bitsAttr(),
        op.quant_minAttr(),
        op.quant_maxAttr(),
        op.use_dynamicAttr(),
        op.round_modeAttr()
    );

    auto output_dynamic_reshape_ty = output_dynamic_reshape_op.getType().cast<RankedTensorType>();
    auto output_dynamic_reshape_out_shape =output_dynamic_reshape_op.output_shape();
    auto new_output_dynamic_reshape_op = rewriter.create<mhlo::DynamicReshapeOp>(
          op->getLoc(),
          output_dynamic_reshape_ty,
          new_output_fake_quant->getResult(0), output_dynamic_reshape_out_shape);
    op->getResult(0).replaceAllUsesWith(new_output_dynamic_reshape_op->getResult(0));
//    input_fake_quant_op->getResult(0).replaceAllUsesWith(new_input_fake_quant->getResult(0));
//    weight_fake_quant_op->getResult(0).replaceAllUsesWith(new_weight_fake_quant->getResult(0));
//    dot_general_op->getResult(0).replaceAllUsesWith(new_dot_op->getResult(0));
    return success();
  }
};

struct FakeQuantDotGeneralDynamicReshapeConverter
    : public OpRewritePattern<mhlo_disc::FakeQuantOp> {
  using OpRewritePattern<mhlo_disc::FakeQuantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo_disc::FakeQuantOp op,
                                PatternRewriter& rewriter) const override {
    std::cout << "match start!!!!\n";
    auto output_dynamic_reshape_op = op.input().template getDefiningOp<mhlo::DynamicReshapeOp>();
    if (!output_dynamic_reshape_op) {
        std::cout << "Not dynamic reshape OP for output\n";
        return failure();
    }
    auto dot_general_op = output_dynamic_reshape_op.operand().template getDefiningOp<mhlo::DotGeneralOp>();
    if (!dot_general_op) {
        std::cout << "Not dot general OP\n";
        return failure();
    }

    auto input_dynamic_reshape_op = dot_general_op.lhs().template getDefiningOp<mhlo::DynamicReshapeOp>();
    if (!input_dynamic_reshape_op) {
        std::cout << "Not dynamic reshape OP for input!\n";
        return failure();
    }

    auto input_fake_quant_op = input_dynamic_reshape_op.operand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!input_fake_quant_op) {
        std::cout << "Not fake quant OP for input!\n";
        return failure();
    }

    auto weight_transpose_op = dot_general_op.rhs().template getDefiningOp<mhlo::TransposeOp>();
    if (!weight_transpose_op) {
        std::cout << "not transpose op for the weight\n";
        return failure();
    }

    auto weight_fake_quant_op = weight_transpose_op.getOperand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!weight_fake_quant_op) {
        std::cout << "no fake quant op for weight\n";
        return failure();
    }

    auto const_op = weight_fake_quant_op.input().template getDefiningOp<mhlo::ConstantOp>();
    if (!const_op) {
        std::cout << "Not constant OP for weight!\n";
        return failure();
    }
    std::cout << "match success\n";

    auto dynamic_reshape_ty = input_dynamic_reshape_op.getType().cast<RankedTensorType>();
    auto dynamic_reshape_out_shape =input_dynamic_reshape_op.output_shape();
    auto new_input_dynamic_reshape_output = rewriter.create<mhlo::DynamicReshapeOp>(
          op->getLoc(),
          dynamic_reshape_ty,
          input_fake_quant_op.input(), dynamic_reshape_out_shape);
    auto new_input_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_input_dynamic_reshape_output.getType(),
        new_input_dynamic_reshape_output,
        input_fake_quant_op.scale(),
        input_fake_quant_op.zero_point(),
        input_fake_quant_op.use_signedAttr(),
        input_fake_quant_op.use_symmetricAttr(),
        input_fake_quant_op.axisAttr(),
        input_fake_quant_op.num_bitsAttr(),
        input_fake_quant_op.quant_minAttr(),
        input_fake_quant_op.quant_maxAttr(),
        input_fake_quant_op.use_dynamicAttr(),
        input_fake_quant_op.round_modeAttr()
    );

    auto new_weight_transpose_op = rewriter.create<mhlo::TransposeOp>(
        op->getLoc(),
        weight_transpose_op.getType(),
        const_op,
        weight_transpose_op.permutationAttr()
    );
    auto permutation_iter = weight_transpose_op.permutationAttr().getValues<int64_t>();
    std::vector<int64_t> permutation_vec;
    for (auto p: permutation_iter) {
        permutation_vec.push_back(p);
    }
    auto origin_axis_iter = weight_fake_quant_op.axisAttr().getValues<int64_t>();
    std::vector<int64_t> origin_axis_vec;
    for (auto a: origin_axis_iter) {
        origin_axis_vec.push_back(a);
    }
    int64_t new_axis;
    for (int64_t i=0; i<permutation_vec.size(); i++) {
        if(permutation_vec[i] == origin_axis_vec[0]) {
            new_axis = i;
            break;
        }
    }
    auto attrTy = RankedTensorType::get({static_cast<int64_t>(1)},
                                      rewriter.getIntegerType(64));
    auto permuteAttr = DenseIntElementsAttr::get(attrTy, {new_axis});
    auto new_weight_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_weight_transpose_op.getType(),
        new_weight_transpose_op,
        weight_fake_quant_op.scale(),
        weight_fake_quant_op.zero_point(),
        weight_fake_quant_op.use_signedAttr(),
        weight_fake_quant_op.use_symmetricAttr(),
        permuteAttr,
        weight_fake_quant_op.num_bitsAttr(),
        weight_fake_quant_op.quant_minAttr(),
        weight_fake_quant_op.quant_maxAttr(),
        weight_fake_quant_op.use_dynamicAttr(),
        weight_fake_quant_op.round_modeAttr()
    );
    auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
      rewriter.getContext(), {}, {},
      {1}, {0});

    auto new_dot_op = rewriter.create<mhlo::DotGeneralOp>(
        op->getLoc(), dot_general_op.getType(),
        new_input_fake_quant, new_weight_fake_quant, dot_dimension_attr, nullptr
    );
//    auto new_dot_op = rewriter.create<mhlo::DotOp>(
//        op->getLoc(), new_input_fake_quant, new_weight_fake_quant, nullptr
//    );
    auto new_output_fake_quant =  rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_dot_op.getType(),
        new_dot_op,
        op.scale(),
        op.zero_point(),
        op.use_signedAttr(),
        op.use_symmetricAttr(),
        op.axisAttr(),
        op.num_bitsAttr(),
        op.quant_minAttr(),
        op.quant_maxAttr(),
        op.use_dynamicAttr(),
        op.round_modeAttr()
    );

    auto output_dynamic_reshape_ty = output_dynamic_reshape_op.getType().cast<RankedTensorType>();
    auto output_dynamic_reshape_out_shape =output_dynamic_reshape_op.output_shape();
    auto new_output_dynamic_reshape_op = rewriter.create<mhlo::DynamicReshapeOp>(
          op->getLoc(),
          output_dynamic_reshape_ty,
          new_output_fake_quant->getResult(0), output_dynamic_reshape_out_shape);
    op->getResult(0).replaceAllUsesWith(new_output_dynamic_reshape_op->getResult(0));
//    input_fake_quant_op->getResult(0).replaceAllUsesWith(new_input_fake_quant->getResult(0));
//    weight_fake_quant_op->getResult(0).replaceAllUsesWith(new_weight_fake_quant->getResult(0));
//    dot_general_op->getResult(0).replaceAllUsesWith(new_dot_op->getResult(0));
    return success();

   }
};





struct FakeQuantDotDynamicReshapeConverter
    : public OpRewritePattern<mhlo_disc::FakeQuantOp> {
  using OpRewritePattern<mhlo_disc::FakeQuantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo_disc::FakeQuantOp op,
                                PatternRewriter& rewriter) const override {
    auto output_dynamic_reshape_op = op.input().template getDefiningOp<mhlo::DynamicReshapeOp>();
    if (!output_dynamic_reshape_op) {
        std::cout << "Not dynamic reshape OP for output\n";
        return failure();
    }
    auto dot_op = output_dynamic_reshape_op.operand().template getDefiningOp<mhlo::DotOp>();
    if (!dot_op) {
        std::cout << "Not dot OP\n";
        return failure();
    }

    auto input_dynamic_reshape_op = dot_op.lhs().template getDefiningOp<mhlo::DynamicReshapeOp>();
    if (!input_dynamic_reshape_op) {
        std::cout << "Not dynamic reshape OP for input!\n";
        return failure();
    }

    auto input_fake_quant_op = input_dynamic_reshape_op.operand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!input_fake_quant_op) {
        std::cout << "Not fake quant OP for input!\n";
        return failure();
    }

     auto weight_transpose_op = dot_op.rhs().template getDefiningOp<mhlo::TransposeOp>();
    if (!weight_transpose_op) {
        std::cout << "not transpose op for the weight\n";
        return failure();
    }

    auto weight_fake_quant_op = weight_transpose_op.getOperand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!weight_fake_quant_op) {
        std::cout << "no fake quant op for weight\n";
        return failure();
    }

    auto const_op = weight_fake_quant_op.input().template getDefiningOp<mhlo::ConstantOp>();
    if (!const_op) {
        std::cout << "Not constant OP for weight!\n";
        return failure();
    }
    std::cout << "match success\n";

    auto dynamic_reshape_ty = input_dynamic_reshape_op.getType().cast<RankedTensorType>();
    auto dynamic_reshape_out_shape =input_dynamic_reshape_op.output_shape();
    auto new_input_dynamic_reshape_output = rewriter.create<mhlo::DynamicReshapeOp>(
          op->getLoc(),
          dynamic_reshape_ty,
          input_fake_quant_op.input(), dynamic_reshape_out_shape);
    auto new_input_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_input_dynamic_reshape_output.getType(),
        new_input_dynamic_reshape_output,
        input_fake_quant_op.scale(),
        input_fake_quant_op.zero_point(),
        input_fake_quant_op.use_signedAttr(),
        input_fake_quant_op.use_symmetricAttr(),
        input_fake_quant_op.axisAttr(),
        input_fake_quant_op.num_bitsAttr(),
        input_fake_quant_op.quant_minAttr(),
        input_fake_quant_op.quant_maxAttr(),
        input_fake_quant_op.use_dynamicAttr(),
        input_fake_quant_op.round_modeAttr()
    );

    auto new_weight_transpose_op = rewriter.create<mhlo::TransposeOp>(
        op->getLoc(),
        weight_transpose_op.getType(),
        const_op,
        weight_transpose_op.permutationAttr()
    );
    auto permutation_iter = weight_transpose_op.permutationAttr().getValues<int64_t>();
    std::vector<int64_t> permutation_vec;
    for (auto p: permutation_iter) {
        permutation_vec.push_back(p);
    }
    auto origin_axis_iter = weight_fake_quant_op.axisAttr().getValues<int64_t>();
    std::vector<int64_t> origin_axis_vec;
    for (auto a: origin_axis_iter) {
        origin_axis_vec.push_back(a);
    }
    int new_axis;
    for (int i=0; i<permutation_vec.size(); i++) {
        if(permutation_vec[i] == origin_axis_vec[0]) {
            new_axis = i;
            break;
        }
    }
    auto attrTy = RankedTensorType::get({static_cast<int64_t>(1)},
                                      rewriter.getIntegerType(64));
    auto permuteAttr = DenseIntElementsAttr::get(attrTy, {new_axis});
    auto new_weight_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_weight_transpose_op.getType(),
        new_weight_transpose_op,
        weight_fake_quant_op.scale(),
        weight_fake_quant_op.zero_point(),
        weight_fake_quant_op.use_signedAttr(),
        weight_fake_quant_op.use_symmetricAttr(),
        permuteAttr,
        weight_fake_quant_op.num_bitsAttr(),
        weight_fake_quant_op.quant_minAttr(),
        weight_fake_quant_op.quant_maxAttr(),
        weight_fake_quant_op.use_dynamicAttr(),
        weight_fake_quant_op.round_modeAttr()
    );
    auto new_dot_op = rewriter.create<mhlo::DotOp>(
        op->getLoc(), new_input_fake_quant, new_weight_fake_quant, nullptr
    );
    auto new_output_fake_quant =  rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_dot_op.getType(),
        new_dot_op,
        op.scale(),
        op.zero_point(),
        op.use_signedAttr(),
        op.use_symmetricAttr(),
        op.axisAttr(),
        op.num_bitsAttr(),
        op.quant_minAttr(),
        op.quant_maxAttr(),
        op.use_dynamicAttr(),
        op.round_modeAttr()
    );

    auto output_dynamic_reshape_ty = output_dynamic_reshape_op.getType().cast<RankedTensorType>();
    auto output_dynamic_reshape_out_shape =output_dynamic_reshape_op.output_shape();
    auto new_output_dynamic_reshape_op = rewriter.create<mhlo::DynamicReshapeOp>(
          op->getLoc(),
          output_dynamic_reshape_ty,
          new_output_fake_quant->getResult(0), output_dynamic_reshape_out_shape);
    op->getResult(0).replaceAllUsesWith(new_output_dynamic_reshape_op->getResult(0));
    return success();

   }
};

struct FakeQuantDotConverter
    : public OpRewritePattern<mhlo::DotOp> {
  using OpRewritePattern<mhlo::DotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter& rewriter) const override {
    auto weight_transpose_op = op.rhs().template getDefiningOp<mhlo::TransposeOp>();
    if (!weight_transpose_op) {
        return failure();
    }
    auto weight_fake_quant_op = weight_transpose_op.getOperand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!weight_fake_quant_op) {
        std::cout << "no fake quant op for weight\n";
        return failure();
    }

    auto const_op = weight_fake_quant_op.input().template getDefiningOp<mhlo::ConstantOp>();
    if (!const_op) {
        std::cout << "Not constant OP for weight!\n";
        return failure();
    }
    auto new_weight_transpose_op = rewriter.create<mhlo::TransposeOp>(
        op->getLoc(),
        weight_transpose_op.getType(),
        const_op,
        weight_transpose_op.permutationAttr()
    );

    // change the axis attr
    auto permutation_iter = weight_transpose_op.permutationAttr().getValues<int64_t>();
    std::vector<int64_t> permutation_vec;
    for (auto p: permutation_iter) {
        permutation_vec.push_back(p);
    }
    auto origin_axis_iter = weight_fake_quant_op.axisAttr().getValues<int64_t>();
    std::vector<int64_t> origin_axis_vec;
    for (auto a: origin_axis_iter) {
        origin_axis_vec.push_back(a);
    }
    int new_axis;
    for (int i=0; i<permutation_vec.size(); i++) {
        if(permutation_vec[i] == origin_axis_vec[0]) {
            new_axis = i;
            break;
        }
    }
    auto attrTy = RankedTensorType::get({static_cast<int64_t>(1)},
                                      rewriter.getIntegerType(64));
    auto permuteAttr = DenseIntElementsAttr::get(attrTy, {new_axis});


    auto new_weight_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_weight_transpose_op.getType(),
        new_weight_transpose_op,
        weight_fake_quant_op.scale(),
        weight_fake_quant_op.zero_point(),
        weight_fake_quant_op.use_signedAttr(),
        weight_fake_quant_op.use_symmetricAttr(),
        permuteAttr,
        weight_fake_quant_op.num_bitsAttr(),
        weight_fake_quant_op.quant_minAttr(),
        weight_fake_quant_op.quant_maxAttr(),
        weight_fake_quant_op.use_dynamicAttr(),
        weight_fake_quant_op.round_modeAttr()
    );
    op.rhs().replaceAllUsesWith(new_weight_fake_quant->getResult(0));
    return success();
  }
};

struct FakeQuantConverter
    : public OpRewritePattern<mhlo_disc::FakeQuantOp> {
  using OpRewritePattern<mhlo_disc::FakeQuantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::FakeQuantOp op,
                                PatternRewriter& rewriter) const override {
    auto output_reshape_op = op.input().template getDefiningOp<mhlo::ReshapeOp>();
    if (!output_reshape_op) {
        std::cout << "Not reshape OP for output\n";
        return failure();

    }
    auto dot_op = output_reshape_op.getOperand().template getDefiningOp<mhlo::DotOp>();
    if (!dot_op) {
        std::cout << "Not dot OP\n";
        return failure();
    }

    auto input_reshape_op = dot_op.lhs().template getDefiningOp<mhlo::ReshapeOp>();
    if (!input_reshape_op) {
        std::cout << "Not reshape OP for input!\n";
        return failure();
    }
    auto input_fake_quant_op = input_reshape_op.getOperand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!input_fake_quant_op) {
        std::cout << "Not fake quant OP for input!\n";
        return failure();
    }
    auto weight_transpose_op = dot_op.rhs().template getDefiningOp<mhlo::TransposeOp>();
    if (!weight_transpose_op) {
        std::cout << "not transpose op for the weight\n";
        return failure();
    }

    auto weight_fake_quant_op = weight_transpose_op.getOperand().template getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!weight_fake_quant_op) {
        std::cout << "no fake quant op for weight\n";
        return failure();
    }

    auto const_op = weight_fake_quant_op.input().template getDefiningOp<mhlo::ConstantOp>();
    if (!const_op) {
        std::cout << "Not constant OP for weight!\n";
        return failure();
    }
    std::cout << "match success\n";
//    return failure();

    // move the input reshape in front of the input fake quant
//    BlockAndValueMapping mapping;
//    mapping.map(input_reshape_op.getOperand(), input_fake_quant_op.input());
//    auto new_reshape_op = rewriter.clone(*input_reshape_op.getOperation(), mapping);

    auto new_input_reshape_output = rewriter.create<mhlo::ReshapeOp>(
          dot_op->getLoc(),
          RankedTensorType::get(
              dot_op.lhs().getType().cast<RankedTensorType>().getShape(),
              dot_op.lhs().getType().cast<RankedTensorType>().getElementType()),
         input_fake_quant_op.input());

    auto new_input_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        dot_op->getLoc(),
        new_input_reshape_output.getType(),
        new_input_reshape_output,
        input_fake_quant_op.scale(),
        input_fake_quant_op.zero_point(),
        input_fake_quant_op.use_signedAttr(),
        input_fake_quant_op.use_symmetricAttr(),
        input_fake_quant_op.axisAttr(),
        input_fake_quant_op.num_bitsAttr(),
        input_fake_quant_op.quant_minAttr(),
        input_fake_quant_op.quant_maxAttr(),
        input_fake_quant_op.use_dynamicAttr(),
        input_fake_quant_op.round_modeAttr()
    );
//    dot_op.lhs().replaceAllUsesWith(new_input_fake_quant->getResult(0));

    auto new_weight_transpose_op = rewriter.create<mhlo::TransposeOp>(
        op->getLoc(),
        weight_transpose_op.getType(),
        const_op,
        weight_transpose_op.permutationAttr()
    );

    // change the axis attr
    auto permutation_iter = weight_transpose_op.permutationAttr().getValues<int64_t>();
    std::vector<int64_t> permutation_vec;
    for (auto p: permutation_iter) {
        permutation_vec.push_back(p);
    }
    auto origin_axis_iter = weight_fake_quant_op.axisAttr().getValues<int64_t>();
    std::vector<int64_t> origin_axis_vec;
    for (auto a: origin_axis_iter) {
        origin_axis_vec.push_back(a);
    }
    int new_axis;
    for (int i=0; i<permutation_vec.size(); i++) {
        if(permutation_vec[i] == origin_axis_vec[0]) {
            new_axis = i;
            break;
        }
    }
    auto attrTy = RankedTensorType::get({static_cast<int64_t>(1)},
                                      rewriter.getIntegerType(64));
    auto permuteAttr = DenseIntElementsAttr::get(attrTy, {new_axis});


    auto new_weight_fake_quant = rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_weight_transpose_op.getType(),
        new_weight_transpose_op,
        weight_fake_quant_op.scale(),
        weight_fake_quant_op.zero_point(),
        weight_fake_quant_op.use_signedAttr(),
        weight_fake_quant_op.use_symmetricAttr(),
        permuteAttr,
        weight_fake_quant_op.num_bitsAttr(),
        weight_fake_quant_op.quant_minAttr(),
        weight_fake_quant_op.quant_maxAttr(),
        weight_fake_quant_op.use_dynamicAttr(),
        weight_fake_quant_op.round_modeAttr()
    );
//    dot_op.rhs().replaceAllUsesWith(new_weight_fake_quant->getResult(0));

    auto new_dot_op = rewriter.create<mhlo::DotOp>(
        op->getLoc(), new_input_fake_quant, new_weight_fake_quant, nullptr
    );


    auto new_output_fake_quant =  rewriter.create<mhlo_disc::FakeQuantOp>(
        op->getLoc(),
        new_dot_op.getType(),
        new_dot_op,
        op.scale(),
        op.zero_point(),
        op.use_signedAttr(),
        op.use_symmetricAttr(),
        op.axisAttr(),
        op.num_bitsAttr(),
        op.quant_minAttr(),
        op.quant_maxAttr(),
        op.use_dynamicAttr(),
        op.round_modeAttr()
    );

    auto new_output_reshape_op = rewriter.create<mhlo::ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get(
              output_reshape_op.getType().cast<RankedTensorType>().getShape(),
              output_reshape_op.getType().cast<RankedTensorType>().getElementType()),
         new_output_fake_quant->getResult(0));
   op->getResult(0).replaceAllUsesWith(new_output_reshape_op->getResult(0));


    return success();
  }

};


struct DiscFoldConstForFakeQuantOpPass
    : public DiscFoldConstForFakeQuantOpPassBase<DiscFoldConstForFakeQuantOpPass> {
        void runOnOperation() override;
};

void populateQuantizedPatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    FakeQuantConverter,
    FakeQuantDotConverter,
    FakeQuantDotDynamicReshapeConverter,
    FakeQuantDotGeneralDynamicReshapeConverter,
    FakeQuantDotGeneralDynamicReshapeConverter2
  >(patterns.getContext());
  // clang-format on
}

void DiscFoldConstForFakeQuantOpPass::runOnOperation() {
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateQuantizedPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscFoldConstForFakeQuantOpPass() {
  return std::make_unique<DiscFoldConstForFakeQuantOpPass>();
}

} // namespace disc_ral
} // namespace mlir