from argparse import ArgumentParser, Namespace

from logging import getLogger

from transformers import AutoModel, AutoTokenizer
from transformers.commands import BaseTransformersCLICommand


def convert_command_factory(args: Namespace):
    return ConvertCommand(args.model_type, args.tf_checkpoint, args.pytorch_dump_output,
                          args.config, args.finetuning_task_name)

