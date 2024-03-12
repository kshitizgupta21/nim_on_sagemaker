from nemollm_inference.sdk.entrypoints.openai import OpenAI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from http import HTTPStatus
from nemollm_inference.api.interfaces.service_provider import ServiceProvider
from typing_extensions import override
from nemollm_inference.api.protocols.openai import CompletionResponse

class HealthResponse(BaseModel):
    object: str = "health-response"
    message: str

class SageMaker(OpenAI):
    def __init__(self, service_provider: ServiceProvider):
        super().__init__(service_provider)

        # SageMaker
        self.app.add_api_route("/ping", self.get_health_ready, methods=["GET"])

        # completion
        self.app.add_api_route(
            "/invocations",
            self.create_completion,
            methods=["POST"],
            response_model_exclude_none=True,
            response_model=CompletionResponse,
        )

    async def get_health_ready(self):
        is_ready = await self._service_provider.is_ready()
        status_code = HTTPStatus.OK if is_ready else HTTPStatus.SERVICE_UNAVAILABLE
        message = "Service is ready." if is_ready else "Service is not ready."
        return self.create_error_response(status_code=status_code, message=message)

    @override
    def create_error_response(self, status_code: HTTPStatus, message: str) -> JSONResponse:
        return JSONResponse(HealthResponse(message=message).model_dump(), status_code=status_code.value)
