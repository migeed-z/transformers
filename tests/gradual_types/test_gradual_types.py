import unittest
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, D
from src.transformers import *
import z3
from timeit import default_timer as timer
from datetime import timedelta
from tests.gradual_types.helper_functions import generate_trace

class HFModels(unittest.TestCase):
    def test_RobertaModel(self):
        print("Roberta Model")
        start = timer()
        trace = generate_trace(RobertaModel)
        end = timer()
        print(len(trace.graph.nodes))
        print(timedelta(seconds=end-start))

    def test_MegatronBertModel(self):
        print("Megatron Bert Model")
        start = timer()
        trace = generate_trace(MegatronBertModel)
        end = timer()
        print(timedelta(seconds=end-start))
        print(len(trace.graph.nodes))

    def test_MobileBertModel(self):
        print("Mobile Bert Model")
        input = z3.Const(1, tensor_type)
        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        user_constraints = z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))])
        start = timer()
        trace = generate_trace(MobileBertModel, user_constraints=user_constraints)
        end = timer()
        print(timedelta(seconds=end-start))
        print(len(trace.graph.nodes))

    def test_BertModel(self):
        print("Bert Model")
        start = timer()
        trace = generate_trace(BertModel)
        end = timer()
        print(timedelta(seconds=end-start))
        print(len(trace.graph.nodes))

    def test_electra_model(self):
        print("Electra Model")
        input = z3.Const(1, tensor_type)
        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        user_constraints = z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))])
        start = timer()
        trace = generate_trace(ElectraModel, user_constraints=user_constraints)
        end = timer()
        print(timedelta(seconds=end-start))
        print(len(trace.graph.nodes))

    # NOTE: this benchmark is time consuming. Please skip if needed.
    def test_xglm(self):
        print('XGLM Model')
        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        input = z3.Const(1, tensor_type)
        # constraints for XGLMModel that say that the input is a tensor of size 2 with the last dimension
        # ranging over a set of natural numbers
        user_constraints_XGLMModel = z3.And([input == tensor_type.tensor2(D(1, s2), D(1, s3)),  s3 > 1, s3 < 2000,
                                             s2 > 0])
        start = timer()
        t = generate_trace(XGLMModel, user_constraints=user_constraints_XGLMModel, hidden_layers=1)
        end = timer()
        print(timedelta(seconds=end-start))
        print(len(t.graph.nodes))
