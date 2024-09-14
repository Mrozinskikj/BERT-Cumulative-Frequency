from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from nlp_engineer_assignment import Tokeniser
import torch

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

        tokeniser = Tokeniser()
        try:
            tokens = tokeniser.encode(text)
            logits = model(tokens) # derive the logits of inputs
            prediction = torch.argmax(logits, dim=-1) # prediction is the highest value logit for each item in sequence
            prediction_string = ''.join(str(i.item()) for i in prediction[0])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        return {"prediction": prediction_string}
    
    return app