from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from nlp_engineer_assignment import Tokeniser, predict
import torch

def process_prediction(
    input_text: str,
    model: 'BERT'
) -> str:
    """
    Prepare the input string for the model, get the model prediction, and convert the prediction tensor to a string.

    Parameters
    ----------
    input_text : str
        The input string from the API request.
    model : BERT
        The BERT model for computing predictions with.
    
    Returns
    -------
    str
        The model prediction converted into a string for API output.
    """
    tokeniser = Tokeniser()
    input_ids = tokeniser.encode(input_text)

    prediction_tensor = predict(input_ids, model)
    prediction_string = ''.join(str(i.item()) for i in prediction_tensor[0])
    return prediction_string


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
    async def serve(input: TextInput) -> dict:
        input_text = input.text
        try:
            prediction = process_prediction(input_text, model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": prediction}
    
    return app