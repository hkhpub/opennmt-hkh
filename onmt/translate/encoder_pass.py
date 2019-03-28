#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import configargparse
import codecs
import os
import math
import numpy as np

import torch

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble


def build_encoder_pass(opt, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')
    dummy_parser = configargparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt, dummy_opt.__dict__)

    encoder_pass = EncoderPass(model, fields, opt, out_file)
    return encoder_pass


class EncoderPass(object):
    def __init__(self, model, fields, opt, outfile):
        self.model = model
        self.fields = fields
        self.data_type = opt.data_type
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1
        self.outfile = outfile

    def encode_seq(self, src, tgt=None, src_dir=None, batch_size=None):
        assert src is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")

        data = inputters.build_dataset(
            self.fields,
            self.data_type,
            src=src,
            tgt=tgt,
            src_dir=src_dir)

        cur_device = "cuda" if self.cuda else "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=cur_device,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        all_sent_vecs = []
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                batch_size = batch.batch_size

                # Encoder forward.
                src, enc_states, memory_bank, src_lengths = self._run_encoder(batch, data.data_type)
                # memory_bank (seq_lengths, batch_size, hidden_size)
                sent_vec_batch = memory_bank.mean(dim=0).cpu().numpy()
                np.savetxt(self.outfile, sent_vec_batch, fmt='%.10e')

                if (i + 1) % 10 == 0:
                    print(".", end="", flush=True)
                if (i + 1) % 100 == 0:
                    print((i + 1)*batch_size, end="", flush=True)

        # results = {}
        # results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        # results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812

    def _run_encoder(self, batch, data_type):
        src = inputters.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths
