from fastapi import FastAPI, Request, Body
from pydantic import BaseModel
from starlette.responses import RedirectResponse


def create_app(model):
    print(model)
    app = FastAPI(
        title="BERT Cumulative Frequency",
        version="1.0.0"
    )

    @app.get("/", include_in_schema=False)
    async def index():
        """
        Redirects to the OpenAPI Swagger UI
        """
        return RedirectResponse(url="/docs")

    class TextInput(BaseModel):
        text: str

    @app.post("/")
    async def predict(input: TextInput) -> dict:
        text = input.text

        prediction = text*2
        
        return {"prediction": prediction}
    
    return app