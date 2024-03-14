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
import asyncio
import multiprocessing
from functools import partial

import gunicorn.app.base
from fastapi import FastAPI
from nemollm_inference.sdk.backends.nim_triton_bls import NimTritonBackendProvider
from nemollm_inference.sdk.common.model import Model
from nemollm_inference.sdk.entrypoints.health import Health
from nemollm_inference.sdk.entrypoints.nemollm import Nemo
from nemollm_inference.sdk.entrypoints.openai import OpenAI
from nemollm_inference.sdk.entrypoints.service import Service
from nemollm_inference.triton.utils.logging import init_json_logging_with_request_id
from nemollm_inference.triton.utils.names import get_full_bls_name, get_trtllm_name_from_directory
from nemollm_inference.sdk.entrypoints.sagemaker import SageMaker

class CustomGunicornApp(gunicorn.app.base.BaseApplication):
    """ 
    This gunicorn app class provides create and exit callbacks for workers, 
    and runs gunicorn with a specified number of workers and multiple gthreads
    """

    def __init__(self, create_app_callback, host_port, num_workers):
        self._configBind = host_port
        self._createAppCallback = create_app_callback
        self.num_workers = num_workers
        super().__init__()

    def load_config(self):
        self.cfg.set("bind", self._configBind)
        self.cfg.set("worker_class", "uvicorn.workers.UvicornWorker")
        self.cfg.set("workers", self.num_workers)
        self.cfg.set("timeout", 200)
        self.cfg.set("loglevel", "info")

    def load(self):
        # This function is invoked when a worker is booted
        self._createdApp = self._createAppCallback()
        return self._createdApp


def main() -> Service:
    parser = argparse.ArgumentParser(
        description='Run FastAPI Service.', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_name', type=str, default='gpt2', help='Name of the model to be loaded.')
    parser.add_argument('--openai_port', type=int, default=9999, help='API port for OpenAI')
    parser.add_argument('--nemo_port', type=int, default=9998, help='API port for NeMo LLM API')
    parser.add_argument(
        '--health_port', type=int, default=8080, help="A port which is used for health checks of the API service."
    )
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Where to host the Open AI API')
    parser.add_argument('--log_level', type=str, default='info', help='Level of logging')
    parser.add_argument('--triton_url', type=str, default='0.0.0.0:8001', help='URL where triton backend is hosted')
    parser.add_argument('--triton_model_name', type=str, default='ensemble', help='Name of triton backend model')
    parser.add_argument('--num_workers', type=str, default=1, help='Number of workers')
    parser.add_argument("--enable_sagemaker", default=False, help="Enable SageMaker support. If enabled, runs server and health check on 8080. Default health check server will run on 8081", action='store_true')


    args = parser.parse_args()
    return run(
        args.model_name,
        args.openai_port,
        args.nemo_port,
        args.host,
        args.log_level,
        args.triton_url,
        args.health_port,
        args.triton_model_name,
        args.enable_sagemaker
    )


def run(
    model_name: str = "gpt2",
    openai_port: int = 9999,
    nemo_port: int = 9998,
    host: str = "0.0.0.0",
    log_level: str = "info",
    triton_url: str = "0.0.0.0:8001",
    health_port: int = 8080,
    triton_model_name: str = "ensemble",
    num_workers: int = 1,
    enable_sagemaker: bool = False
):

    init_json_logging_with_request_id(log_level.upper())
    triton_model_name = get_full_bls_name(triton_model_name)
    trtllm_model_name = get_trtllm_name_from_directory(directory="/model-store/")

    models = [
        Model(
            model_name=model_name,
            backend_provider=NimTritonBackendProvider(
                url=triton_url,
                public_model_name=model_name,
                triton_model_name=triton_model_name,
                trtllm_model_name=trtllm_model_name,
            ),
        )
    ]
    service = Service(models=models, host=host, num_workers=num_workers)
    if enable_sagemaker:
       # Note: SM has the same port 8080 for health and invocations. Nvidia team to figure out how to register endpoint on same port
       service.register_entrypoints(entrypoints_class=SageMaker, port=f"{host}:{str(health_port)}")
       
    service.register_entrypoints(entrypoints_class=Nemo, port=f"{host}:{str(nemo_port)}")
    service.register_entrypoints(entrypoints_class=OpenAI, port=f"{host}:{str(openai_port)}")
    if not enable_sagemaker:
        service.register_entrypoints(entrypoints_class=Health, port=f"{host}:{str(health_port)}")
    asyncio.run(initialize(service))
    processes = []
    for port, app in service._entrypoints.items():
        create_app_callback = partial(get_app_callback, service, port, isinstance(app, Health))
        server = CustomGunicornApp(create_app_callback, port, service.num_workers)
        process = multiprocessing.Process(target=server.run)
        processes.append(process)
        process.start()
    try:
        # Keep the script running until interrupted
        for process in processes:
            process.join()
    except Exception as e:
        # Terminate all processes if interrupted
        for process in processes:
            process.terminate()
            process.join()
    return service


async def initialize(service: Service) -> None:
    await service.initialize()


async def run_service(service: Service) -> None:
    await service.run()


def get_app_callback(service: Service, port: str, is_health_app: bool = False) -> FastAPI:
    # only await backend for non-health endpoints
    if not is_health_app:
        asyncio.run(run_service(service))
    return service._entrypoints[port].app


if __name__ == "__main__":
    main()
