#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import multiprocessing
import os
import subprocess
import threading

from nemollm_inference.service.entrypoints_from_cli import run
from nemollm_inference.triton.utils.logging import init_json_logging_with_request_id


def validate_num_gpus(num_gpus):
    err = False
    n = None
    try:
        n = int(num_gpus)
        if n <= 0:
            err = True
    except Exception:
        err = True

    if err:
        raise ValueError(f"invalid num_gpus {num_gpus}")
    return n


parser = argparse.ArgumentParser(
    description="NeMo Microservice Inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--model_name", help="Specific model to be deployed", required=True)
parser.add_argument("--nemo_port", type=int, help="Port at which the NeMo LLM API runs on", default="9998")
parser.add_argument("--openai_port", type=int, help="Port at which the OpenAI API runs on", default="9999")
parser.add_argument(
    "--num_gpus", type=validate_num_gpus, help="Number of GPU's specified, if not provided the default", required=True
)
parser.add_argument('--host', type=str, default='0.0.0.0', help='Where to host the Open AI API')
parser.add_argument('--log_level', type=str, default='debug', help='Level of logging')
parser.add_argument('--triton_url', type=str, default='0.0.0.0:8001', help='URL where triton backend is hosted')
parser.add_argument(
    '--health_port', default=8080, type=int, help="A port which is used for health checks of the API service."
)
parser.add_argument('--triton_model_name', type=str, default='ensemble', help='Name of triton backend model')
parser.add_argument(
    '--data_store_url',
    default="gateway-api:9009",
    help="An URL of a data store service. A data store service can be "
    "either Model Management Service or Managed NeMo LLM service.",
)
parser.add_argument(
    '--customization_source',
    default="MMS",
    choices=["MMS", "SERVICE"],
    help="A type of a model service which URL is provided in --data_store_url. MMS "
    "stands for Model Management Service and SERVICE stands for Managed NeMo LLM service.",
)
parser.add_argument('--num_workers', type=str, default=1, help='Number of workers')
parser.add_argument("--enable_sagemaker", help="Enable SageMaker support. If enabled, runs server and health check on 8080. Default health check server will run on 8081", action='store_true', default=False)

def run_command(command):
    proc = subprocess.Popen(command)
    proc.wait()
    return proc.returncode


def run_cli(args):
    p = multiprocessing.Process(
        target=run,
        args=(
            args.model_name,
            args.openai_port,
            args.nemo_port,
            args.host,
            args.log_level,
            args.triton_url,
            args.health_port,
            args.triton_model_name,
            args.num_workers,
            args.enable_sagemaker
        ),
    )
    p.start()
    p.join()
    return p.exitcode


def main():
    args = parser.parse_args()
    os.environ["DATA_STORE_URL"] = args.data_store_url
    os.environ["CUSTOMIZATION_SOURCE"] = args.customization_source
    init_json_logging_with_request_id(args.log_level.upper())

    num_gpus_str = str(args.num_gpus)
    triton_command = [
        "python3",
        "/opt/nemollm/launch_trt_triton_server.py",
        "--world_size",
        num_gpus_str,
        "--allow-run-as-root"
    ]

    t1 = threading.Thread(target=run_cli, args=(args,), daemon=True)
    t1.start()
    triton_return_code = run_command(triton_command)
    if triton_return_code != 0:
        print(f"Triton command return code: {triton_return_code}")


if __name__ == '__main__':
    main()
