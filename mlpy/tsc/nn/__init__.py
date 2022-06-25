from .inception_v4 import InceptionNetV4

from .encoder import Encoder
from .fcn import FCN
from .mcdcnn import MCDCNN
from .mcnn import MCNN
from .mlp import MLP
from .resnet import ResNet
from .tcnn import TCNN
from .tlenet import TLeNet
from .twiesn import TWIESN
from .tscnn import TSCNN
FawazReview2019 = {
    'mlp': MLP,
    'fcn': FCN,
    'resnet': ResNet,
    'encoder': Encoder,
    'mcnn': MCNN,
    'tlenet': TLeNet,
    'mcdcnn': MCDCNN,
    'tcnn': TCNN,
    'twiesn': TWIESN,
    'tscnn': TSCNN
}

from .lstmfcn import LSTMFCN
from .lstmfcn import LSTMFCNAttention
from .lstmfcnaug import LSTMFCNAug
from .lstmfcnaug import LSTMFCNAugAttention

MODELS_REALIZED_BY_KERAS = {LSTMFCN,  LSTMFCNAttention, LSTMFCNAug, LSTMFCNAugAttention}

