import inspect
import unittest
import torch
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_all_constraints
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, D
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx import GraphModule
from enum import Enum
from torch.fx.tensor_type import TensorType as TT
from torch.fx.tensor_type import Dyn
from src.transformers import *
import src.transformers.utils.fx as fx
import z3

bs = 4
num_choices = 3
seq_length = 32


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2



def generate_concrete_args_for_model(model, input_names=None):
    input_names = input_names if input_names else model.dummy_inputs.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    return concrete_args


def generate_hf_model(model_cls, hidden_layers=None):
    config_cls = model_cls.config_class
    config = config_cls()

    # we simplify the model for now by removing the hidden layers
    if hidden_layers is not None:
        config.num_hidden_layers = hidden_layers
    if model_cls in [GPT2ForSequenceClassification, GPTNeoForSequenceClassification, GPTJForSequenceClassification] or \
            model_cls.__name__.startswith("Roberta") or model_cls.__name__.startswith("Marian"):
        config.pad_token_id = 0
    model = model_cls(config)
    model.eval()

    return model


def generate_inputs_for_model(model_cls, model, include_loss_args=False):
    if model_cls.__name__.endswith('MultipleChoice'):
        input = torch.zeros(bs, num_choices, seq_length, dtype=torch.long).random_(model.config.vocab_size)
    elif model_cls.__name__.startswith("Roberta"):
        input = torch.zeros(bs, seq_length, dtype=torch.long)
    else:
        input = torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size)

    if 'Bart' in model_cls.__name__:
        input[:, -1] = model.config.eos_token_id

    input_dict = {'input_ids': input}

    if model_cls.__name__.startswith("T5") or model_cls.__name__.startswith("M2M100") \
            or model_cls.__name__.startswith("MT5") or model_cls in [BlenderbotModel, BlenderbotSmallModel,
                                                                     BlenderbotForConditionalGeneration,
                                                                     BlenderbotSmallForConditionalGeneration,
                                                                     PegasusModel, PegasusForConditionalGeneration,
                                                                     MarianModel, MarianMTModel]:
        input_dict.update({'decoder_input_ids': input})

    if include_loss_args:
        if model_cls.__name__.endswith('PreTraining'):
            if model_cls == ElectraForPreTraining:
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(1),
                })
            else:
                label_name = 'sentence_order_label' if model_cls in [AlbertForPreTraining] else 'next_sentence_label'
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
                    label_name: torch.zeros(bs, dtype=torch.long).random_(1),
                })
        elif model_cls.__name__.endswith('QuestionAnswering'):
            input_dict.update({
                'start_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length),
                'end_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length)
            })
        elif (model_cls.__name__.endswith('MaskedLM') or model_cls.__name__.endswith('HeadModel') or
              model_cls.__name__.endswith('CausalLM') or model_cls.__name__.endswith('DoubleHeadsModel')):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
            })
        elif model_cls.__name__.endswith('TokenClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('MultipleChoice'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(num_choices),
            })
        elif model_cls.__name__.endswith('SequenceClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('NextSentencePrediction'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(1),
            })
        elif model_cls.__name__.endswith('ForConditionalGeneration'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size - 1),
            })
        else:
            raise NotImplementedError(f'Class {model_cls.__name__} unsupported for training test ')

    return input_dict


model_classes = [XGLMModel, AlbertModel, BartModel, BertModel, DistilBertModel, ElectraModel, GPT2Model,
 GPTJModel, GPTNeoModel, MegatronBertModel, MobileBertModel, RobertaModel, T5Model,
 BlenderbotModel, BlenderbotSmallModel]


def generate_trace(model_class, user_constraints=None, hidden_layers=None):
    m = generate_hf_model(model_class, hidden_layers)
    input_dict = generate_inputs_for_model(model_class, m)
    concrete_args = generate_concrete_args_for_model(m, input_dict.keys())
    hf_tracer = fx.HFTracer(user_constraints=user_constraints)
    g = GraphModule(m, hf_tracer.trace(m, concrete_args=concrete_args))
    return g


class HFModels(unittest.TestCase):

    def test_electra_model_0(self):
        input = z3.Const(1, tensor_type)
        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        user_constraints = z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2))])
        trace = generate_trace(ElectraModel, hidden_layers=1, user_constraints=user_constraints)

    def test_trace_model_hidden_layers_2(self):
        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        input = z3.Const(1, tensor_type)

        # constraints for XGLMModel that say that the input is a tensor of size 2 with the last dimension
        # ranging over a set of natural numbers
        user_constraints_XGLMModel = z3.And([input == tensor_type.tensor2(D(1, s2), D(1, s3)),  s3 > 1, s3 < 2000,
                                             s2 > 0])
        generate_trace(XGLMModel, user_constraints=user_constraints_XGLMModel, hidden_layers=2)


    def test_trace_model_hidden_layers_1(self):

        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        input = z3.Const(1, tensor_type)

        # constraints for XGLMModel that say that the input is a tensor of size 2 with the last dimension
        # ranging over a set of natural numbers
        user_constraints_XGLMModel = z3.And([input == tensor_type.tensor2(D(1, s2), D(1, s3)), s3 > 1, s3 < 2000, s2 > 0])
        generate_trace(XGLMModel, user_constraints=user_constraints_XGLMModel, hidden_layers=1)


    def test_trace_model(self):

        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        input = z3.Const(1, tensor_type)

        # constraints for XGLMModel that say that the input is a tensor of size 2 with the last dimension
        # ranging over a set of natural numbers
        user_constraints_XGLMModel = z3.And([input == tensor_type.tensor2(D(s1, s2), D(1, s3)),  s3 > 1, s3 < 2000])

        XGLMModel_trace = generate_trace(XGLMModel, user_constraints=user_constraints_XGLMModel, hidden_layers=0)

        input = torch.ones([4, 32], dtype=torch.long)
        # generate shapes for a particular input to compare with
        # our shape inference
        sample_input = input
        ShapeProp(XGLMModel_trace).propagate(sample_input)

        for n in XGLMModel_trace.graph.nodes:
            if n.target == 'layer_norm':
                layer_norm_size = n.meta['tensor_meta'].shape

        for n in XGLMModel_trace.graph.nodes:
            if n.name == 'input_ids':
                n.type = TT([4, 32])

        constraints = transform_all_constraints(XGLMModel_trace, counter=0)
        s = z3.Solver()
        s.add(constraints)
        self.assertEqual(s.check(), z3.sat)

        layer_norm = z3.Const(85, tensor_type)

        self.assertEqual(s.model()[layer_norm].arg(0).arg(1), layer_norm_size[0])
        self.assertEqual(s.model()[layer_norm].arg(1).arg(1), layer_norm_size[1])
        self.assertEqual(s.model()[layer_norm].arg(2).arg(1), layer_norm_size[2])

        input = torch.ones([4, 1000], dtype=torch.long)
        # generate shapes for a particular input to compare with
        # our shape inference
        sample_input = input
        ShapeProp(XGLMModel_trace).propagate(sample_input)

        for n in XGLMModel_trace.graph.nodes:
            if n.target == 'layer_norm':
                layer_norm_size = n.meta['tensor_meta'].shape

        for n in XGLMModel_trace.graph.nodes:
            if n.name == 'input_ids':
                n.type = TT([4, 1000])

        constraints = transform_all_constraints(XGLMModel_trace, counter=0)
        s = z3.Solver()
        s.add(constraints)
        self.assertEqual(s.check(), z3.sat)
        layer_norm = z3.Const(85, tensor_type)

        self.assertEqual(s.model()[layer_norm].arg(0).arg(1), layer_norm_size[0])
        self.assertEqual(s.model()[layer_norm].arg(1).arg(1), layer_norm_size[1])
        self.assertEqual(s.model()[layer_norm].arg(2).arg(1), layer_norm_size[2])

        input = torch.ones([4, 500], dtype=torch.long)
        # generate shapes for a particular input to compare with
        # our shape inference
        sample_input = input
        ShapeProp(XGLMModel_trace).propagate(sample_input)

        for n in XGLMModel_trace.graph.nodes:
            if n.target == 'layer_norm':
                layer_norm_size = n.meta['tensor_meta'].shape

        for n in XGLMModel_trace.graph.nodes:
            if n.name == 'input_ids':
                n.type = TT([4, 500])

        constraints = transform_all_constraints(XGLMModel_trace, counter=0)
        s = z3.Solver()
        s.add(constraints)
        self.assertEqual(s.check(), z3.sat)
        layer_norm = z3.Const(85, tensor_type)

        self.assertEqual(s.model()[layer_norm].arg(0).arg(1), layer_norm_size[0])
        self.assertEqual(s.model()[layer_norm].arg(1).arg(1), layer_norm_size[1])
        self.assertEqual(s.model()[layer_norm].arg(2).arg(1), layer_norm_size[2])


    def test_trace_model_r3(self):

        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        input = z3.Const(1, tensor_type)

        # constraints for XGLMModel that say that the input is a tensor of size 2 with the last dimension
        # ranging over a set of natural numbers
        user_constraints_XGLMModel = z3.And([input == tensor_type.tensor3(D(s1, s2), D(s2, s3), D(1, s5)),  s5 > 1, s5 < 2000])

        XGLMModel_trace = generate_trace(XGLMModel, user_constraints=user_constraints_XGLMModel, hidden_layers=0)
        input = torch.ones([4, 4, 32], dtype=torch.long)
        # generate shapes for a particular input to compare with
        # our shape inference
        sample_input = input
        ShapeProp(XGLMModel_trace).propagate(sample_input)

        for n in XGLMModel_trace.graph.nodes:
            if n.target == 'layer_norm':
                layer_norm_size = n.meta['tensor_meta'].shape

        for n in XGLMModel_trace.graph.nodes:
            if n.name == 'input_ids':
                n.type = TT([4, 4, 32])

        constraints = transform_all_constraints(XGLMModel_trace, counter=0)
        s = z3.Solver()
        s.add(constraints)
        self.assertEqual(s.check(), z3.sat)

        layer_norm = z3.Const(85, tensor_type)

        self.assertEqual(s.model()[layer_norm].arg(0).arg(1), layer_norm_size[0])
        self.assertEqual(s.model()[layer_norm].arg(1).arg(1), layer_norm_size[1])
        self.assertEqual(s.model()[layer_norm].arg(2).arg(1), layer_norm_size[2])


    def test_trace_model_r4(self):

        s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
        input = z3.Const(1, tensor_type)

        # constraints for XGLMModel that say that the input is a tensor of size 2 with the last dimension
        # ranging over a set of natural numbers
        user_constraints_XGLMModel = z3.And([input == tensor_type.tensor4(D(1, s2), D(1, s3), D(1, s4), D(1, s5)),  s5 > 1, s5 < 2000])

        XGLMModel_trace = generate_trace(XGLMModel, user_constraints=user_constraints_XGLMModel, hidden_layers=0)
        input = torch.ones([4, 4, 4, 32], dtype=torch.long)
        # generate shapes for a particular input to compare with
        # our shape inference
        sample_input = input
        ShapeProp(XGLMModel_trace).propagate(sample_input)

        for n in XGLMModel_trace.graph.nodes:
            if n.target == 'layer_norm':
                layer_norm_size = n.meta['tensor_meta'].shape

        for n in XGLMModel_trace.graph.nodes:
            if n.name == 'input_ids':
                n.type = TT([Dyn, 4, 4, 32])

        constraints = transform_all_constraints(XGLMModel_trace, counter=0)
        s = z3.Solver()
        s.add(constraints)
        self.assertEqual(s.check(), z3.sat)

        layer_norm = z3.Const(85, tensor_type)

        self.assertEqual(s.model()[layer_norm].arg(1).arg(1), layer_norm_size[1])
        self.assertEqual(s.model()[layer_norm].arg(2).arg(1), layer_norm_size[2])
