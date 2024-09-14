from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from nlp_engineer_assignment import Tokeniser, predict


def create_app(model):
    """
    Create and configure an instance of the FastAPI application.

    Parameters
    ----------
    model : BERT
        The BERT model for computing predictions with.

    Returns
    -------
    FastAPI
        An instance of the configured FastAPI application.
    """
    app = FastAPI(
        title="BERT Cumulative Frequency",
        version="1.0.0"
    )

    tokeniser = Tokeniser() # create an instance of the tokeniser for input processing

    @app.get("/", include_in_schema=False)
    async def index():
        """
        Redirects to the OpenAPI Swagger UI
        """
        return RedirectResponse(url="/docs")

    class TextInput(BaseModel):
        """
        Schema for the input text received by the API.
        """
        text: str

    @app.post("/")
    async def serve(input: TextInput) -> dict:
        """
        Endpoint to process input text and return model prediction.

        Parameters
        ----------
        input : TextInput
            The input data containing the text to be classified.

        Returns
        -------
        dict
            A dictionary containing the model prediction.

        Raises
        ------
        HTTPException
            If there is a ValueError when unpermitted string is processed by the tokeniser.
        """
        input_text = input.text
        try:
            input_ids = tokeniser.encode(input_text) # encode the input string into tokens
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        prediction_tensor = predict(input_ids, model) # get model predictions from input
        prediction = ''.join(str(i.item()) for i in prediction_tensor[0]) # convert tensor into string
        return {"prediction": prediction}
    
    return app