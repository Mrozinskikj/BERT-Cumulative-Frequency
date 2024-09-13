from fastapi import FastAPI, Request, Body
from starlette.responses import RedirectResponse


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

@app.post("/predict")
async def predict(data: dict = Body(...)):
    text = data.get('text')
    if text is None:
        return {"error": "No text provided"}
    
    prediction = text
    return {"prediction": prediction*2}

# TODO: Add a route to the API that accepts a text input and uses the trained
# model to predict the number of occurrences of each letter in the text up to
# that point.
