# -*- coding: utf-8 -*-
import io
from os.path import abspath, dirname

import PIL
import pytest
import requests

from rudalle import get_tokenizer, get_rudalle_model, get_vae, get_realesrgan, get_emojich_unet


TEST_ROOT = dirname(abspath(__file__))


@pytest.fixture(scope='module')
def realesrgan():
    yield get_realesrgan('x2', device='cpu')


@pytest.fixture(scope='module')
def vae():
    yield get_vae(pretrained=False)


@pytest.fixture(scope='module')
def pretrained_vae():
    yield get_vae()


@pytest.fixture(scope='module')
def dwt_vae():
    yield get_vae(pretrained=False, dwt=True)


@pytest.fixture(scope='module')
def yttm_tokenizer():
    yield get_tokenizer()


@pytest.fixture(scope='module')
def sample_image():
    url = 'https://cdn.kqed.org/wp-content/uploads/sites/12/2013/12/rudolph.png'
    resp = requests.get(url)
    resp.raise_for_status()
    yield PIL.Image.open(io.BytesIO(resp.content))


@pytest.fixture(scope='module')
def sample_image_cat():
    yield PIL.Image.open('pics/ginger_cat.jpeg')


@pytest.fixture(scope='module')
def small_dalle():
    yield get_rudalle_model('dummy', pretrained=False, fp16=False, device='cpu')


@pytest.fixture(scope='module')
def xl_dalle():
    yield get_rudalle_model('Malevich', pretrained=True, fp16=False, device='cpu')


@pytest.fixture(scope='module')
def xxl_dalle():
    yield get_rudalle_model(
        'Kandinsky',
        pretrained=False,
        fp16=False,
        device='cpu',
        cogview_layernorm_prescale=True,
        custom_relax=True,
    )


@pytest.fixture(scope='module')
def emojich_unet():
    yield get_emojich_unet('unet_effnetb5')
