#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse

from onmt.utils.logging import init_logger
from onmt.translate.encoder_pass import build_encoder_pass

import onmt.opts as opts

#
# -gpu 1 -batch_size 100 -model /home/hkh/data/ted2013/opennmt.data.ko-en/models/data.ko-en_step_20000.pt -src /home/hkh/data/ted2013/data.ko-en/tst2017.en-ko.tok.bpe32k.ko -output /tmp/output1.txt
# -gpu 1 -batch_size 100 -model /home/hkh/data/ted2013/opennmt.data.ko-en/models/data.ko-en_step_20000.pt -src /home/hkh/data/ted2013/data.ko-en/train.ko-en.tok.clean.bpe32k.ko -output /tmp/output1.txt
# /home/hkh/data/ted2013/opennmt.data.bilingual.ko-en/models/data.bilingual.ko-en_step_15000.pt


def main(opt):
    encoder_pass = build_encoder_pass(opt)
    encoder_pass.encode_seq(
        src=opt.src,
        tgt=opt.tgt,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
