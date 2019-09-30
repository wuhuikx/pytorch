import torch
import torch.jit
import unittest
from common_utils import run_tests
from common_quantization import QuantizationTestCase, ModelMultipleOps, ModelMultipleOpsNoAvgPool

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     "Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
class ModelNumerics(QuantizationTestCase):
    def test_float_quant_compare_per_tensor(self):
        torch.manual_seed(42)
        my_model = ModelMultipleOps().to(torch.float32)
        my_model.eval()
        calib_data = torch.rand(1024, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(1, 3, 15, 15, dtype=torch.float32)
        out_ref = my_model(eval_data)
        q_model = torch.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = torch.quantization.default_qconfig
        torch.quantization.fuse_modules(q_model.module, [['conv1', 'bn1', 'relu1']])
        torch.quantization.prepare(q_model)
        q_model(calib_data)
        torch.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
        # output
        self.assertGreater(SQNRdB, 30, msg='Quantized model numerics diverge from float, expect SQNR > 30 dB')

    def test_float_quant_compare_per_channel(self):
        # Test for per-channel Quant
        torch.manual_seed(67)
        my_model = ModelMultipleOps().to(torch.float32)
        my_model.eval()
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        out_ref = my_model(eval_data)
        q_model = torch.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = torch.quantization.default_per_channel_qconfig
        torch.quantization.fuse_modules(q_model.module, [['conv1', 'bn1', 'relu1']])
        torch.quantization.prepare(q_model)
        q_model(calib_data)
        torch.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')

    def test_fake_quant_true_quant_compare(self):
        torch.manual_seed(67)
        myModel = ModelMultipleOpsNoAvgPool().to(torch.float32)
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        myModel.eval()
        out_ref = myModel(eval_data)
        fqModel = torch.quantization.QuantWrapper(myModel)
        fqModel.train()
        fqModel.qconfig = torch.quantization.default_qat_qconfig
        torch.quantization.fuse_modules(fqModel.module, [['conv1', 'bn1', 'relu1']])
        torch.quantization.prepare_qat(fqModel)
        fqModel.eval()
        fqModel.apply(torch.quantization.disable_fake_quant)
        fqModel.apply(torch.nn._intrinsic.qat.freeze_bn_stats)
        fqModel(calib_data)
        fqModel.apply(torch.quantization.enable_fake_quant)
        fqModel.apply(torch.quantization.disable_observer)
        out_fq = fqModel(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_fq))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')
        torch.quantization.convert(fqModel)
        out_q = fqModel(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_fq) / (torch.norm(out_fq - out_q) + 1e-10))
        self.assertGreater(SQNRdB, 60, msg='Fake quant and true quant numerics diverge, expect SQNR > 60 dB')

    # Test to compare weight only quantized model numerics and
    # activation only quantized model numerics with float
    def test_weight_only_activation_only_fakequant(self):
        torch.manual_seed(67)
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        qconfigset = set([torch.quantization.default_weight_only_quant_qconfig,
                          torch.quantization.default_activation_only_quant_qconfig])
        SQNRTarget = [35, 45]
        for idx, qconfig in enumerate(qconfigset):
            myModel = ModelMultipleOpsNoAvgPool().to(torch.float32)
            myModel.eval()
            out_ref = myModel(eval_data)
            fqModel = torch.quantization.QuantWrapper(myModel)
            fqModel.train()
            fqModel.qconfig = qconfig
            torch.quantization.fuse_modules(fqModel.module, [['conv1', 'bn1', 'relu1']])
            torch.quantization.prepare_qat(fqModel)
            fqModel.eval()
            fqModel.apply(torch.quantization.disable_fake_quant)
            fqModel.apply(torch.nn._intrinsic.qat.freeze_bn_stats)
            fqModel(calib_data)
            fqModel.apply(torch.quantization.enable_fake_quant)
            fqModel.apply(torch.quantization.disable_observer)
            out_fq = fqModel(eval_data)
            SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_fq))
            self.assertGreater(SQNRdB, SQNRTarget[idx], msg='Quantized model numerics diverge from float')

if __name__ == "__main__":
    run_tests()
