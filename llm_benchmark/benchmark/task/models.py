from pydantic import BaseModel


class EmailCategorizationResponse(BaseModel):
    S: list[int]
    C: list[int]


class EmailTemplateExtractionResponseItem(BaseModel):
    m: int
    j: list[str]
    p: list[str]


class EmailTemplateExtractionResponse(BaseModel):
    result: list[EmailTemplateExtractionResponseItem]


# The Registry
RESPONSE_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "email_categorization_v1": EmailCategorizationResponse,
    "email_template_extraction_v1": EmailTemplateExtractionResponse,
}
